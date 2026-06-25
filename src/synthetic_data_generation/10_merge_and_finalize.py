# 10_merge_and_finalize.py
"""
Merge real panel rows + CTAB-GAN synthetic rows + TimeGAN synthetic rows
into a single merged_full.parquet.

Also writes:
  schema/column_manifest.json
  schema/validation_report.json
"""

import modal
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional

# ── Image ─────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "pandas==2.2.0",
        "numpy==1.26.4",
        "pyarrow==15.0.0",
    ])
    .add_local_file("config.py", "/root/config.py")
    .add_local_file("utils.py",  "/root/utils.py")
)

# ── App & volumes ─────────────────────────────────────────────────────────────
from config import VOLUME_NAMES

app        = modal.App("taxxx-pipeline-10-merge")
final_vol  = modal.Volume.from_name(VOLUME_NAMES["final"],   create_if_missing=True)
gan_vol    = modal.Volume.from_name(VOLUME_NAMES["gan_out"], create_if_missing=True)
merged_vol = modal.Volume.from_name(VOLUME_NAMES["merged"],  create_if_missing=True)
logs_vol   = modal.Volume.from_name(VOLUME_NAMES["logs"],    create_if_missing=True)

VOLUMES = {
    "/final_dataset": final_vol,   # train/val/test live here
    "/gan_output":    gan_vol,     # CTAB + TimeGAN outputs live here
    "/merged":        merged_vol,  # merged_full.parquet written here
    "/logs":          logs_vol,
}

# ── Training year weights for CTAB year assignment ────────────────────────────
_TRAIN_YEARS        = list(range(2019, 2024))
_TRAIN_YEAR_WEIGHTS = np.array(
    [360_000, 16_500, 17_200, 15_800, 16_900], dtype=float
)
_TRAIN_YEAR_WEIGHTS /= _TRAIN_YEAR_WEIGHTS.sum()

# ── Canonical column order ────────────────────────────────────────────────────
CANONICAL_COLUMNS: list[str] = [
    # Identity
    "person_id", "tax_year", "data_source", "zone", "age", "sex",
    "education", "filing_status", "taxpayer_type", "primary_occupation",
    "entry_cohort", "first_year_filing", "employer_id",
    # W-2
    "w2_wages", "federal_withheld", "fica_withheld", "medicare_withheld",
    "federal_withheld_total",
    # Schedule C
    "gross_receipts", "cogs", "gross_profit", "total_expenses",
    "net_se_income", "has_schedule_c",
    "sch_c_advertising", "sch_c_car_truck", "sch_c_depreciation",
    "sch_c_insurance", "sch_c_meals", "sch_c_office_expense",
    "sch_c_rent", "sch_c_repairs", "sch_c_supplies",
    "sch_c_utilities", "sch_c_wages", "sch_c_home_office",
    "se_tax_amount", "se_tax_deduction",
    # Gig
    "gig_income", "gig_expenses", "gig_net", "has_gig",
    # Rental
    "rental_gross", "rental_expenses", "rental_depreciation",
    "rental_net", "has_rental", "n_rental_units",
    # Investment
    "dividends", "capital_gains_lt", "capital_gains_st",
    "interest_income", "has_investments",
    # Crypto
    "crypto_proceeds", "crypto_cost_basis", "crypto_net_gain", "has_crypto",
    # Foreign
    "foreign_income", "foreign_account_balance",
    "fbar_required", "has_foreign_account",
    # K-1
    "k1_income", "owner_salary",
    # Other income
    "social_security_income", "pension_income",
    "ira_distributions", "unemployment_comp",
    # Deductions
    "uses_itemized", "standard_deduction", "itemized_total",
    "itemized_mortgage_int", "itemized_salt", "itemized_charitable",
    "itemized_medical", "deduction_taken", "qbi_deduction",
    # Tax
    "agi", "taxable_income", "tax_before_credits",
    "eitc_credit", "child_tax_credit", "total_credits",
    "total_tax_liability", "effective_tax_rate",
    "refund_amount", "balance_due",
    # Utility / lifestyle
    "electricity_kwh", "gas_therms", "water_gallons",
    "utility_cost_estimated", "utility_income_ratio",
    "lifestyle_income_ratio", "bank_deposit_ratio",
    "deduction_income_ratio", "effective_rate_vs_zone", "irs_risk_score",
    # Policy
    "received_ppp", "ppp_loan_amount",
    # Fraud labels — targets only, no leakage cols
    "fraud_label", "fraud_type", "fraud_category",
]

INT_COLS: frozenset[str] = frozenset([
    "zone", "age", "has_schedule_c", "has_gig", "has_rental",
    "has_investments", "has_crypto", "has_foreign_account",
    "uses_itemized", "first_year_filing", "received_ppp",
    "fbar_required", "fraud_label",
])

STRING_COLS: frozenset[str] = frozenset([
    "person_id", "tax_year", "data_source", "sex", "education",
    "filing_status", "taxpayer_type", "primary_occupation",
    "employer_id", "entry_cohort", "fraud_type", "fraud_category",
])

FLOAT_COLS: frozenset[str] = frozenset([
    c for c in CANONICAL_COLUMNS
    if c not in INT_COLS and c not in STRING_COLS
])

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _align_to_schema(
    df:         pd.DataFrame,
    source_tag: str,
    rng:        np.random.Generator,
) -> pd.DataFrame:
    df = df.copy()

    # Source tag
    df["data_source"] = source_tag

    # tax_year — assign where missing (CTAB has no year column)
    if "tax_year" not in df.columns:
        df["tax_year"] = rng.choice(
            _TRAIN_YEARS, size=len(df), p=_TRAIN_YEAR_WEIGHTS
        )
    else:
        # Fix: only fill the rows that are actually null
        missing_mask = df["tax_year"].isna()
        if missing_mask.any():
            fills = rng.choice(
                _TRAIN_YEARS,
                size=int(missing_mask.sum()),
                p=_TRAIN_YEAR_WEIGHTS,
            )
            df.loc[missing_mask, "tax_year"] = fills

    # person_id — prefix for global uniqueness
    prefix = {
        "real":              "R",
        "ctab_w2":           "CW",
        "ctab_se":           "CS",
        "ctab_itemizers":    "CI",
        "timegan_compliant": "TC",
        "timegan_evader":    "TE",
    }.get(source_tag, "XX")

    if "person_id" not in df.columns:
        df["person_id"] = [f"{prefix}_{i:08d}" for i in range(len(df))]
    else:
        df["person_id"] = prefix + "_" + df["person_id"].astype(str)

    # Add missing canonical columns as None
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Keep only canonical columns
    df = df[CANONICAL_COLUMNS].copy()

    # Enforce int dtypes
    for col in INT_COLS:
        if col in df.columns and df[col].notna().any():
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                  .round()
                  .astype("Int64")
            )

    # Enforce float dtypes
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _validate(df: pd.DataFrame, sources: dict, log) -> dict:
    report: dict = {}
    report["total_rows"]       = len(df)
    report["source_breakdown"] = sources

    fraud_rate = float(df["fraud_label"].fillna(0).mean())
    report["fraud_rate_overall"] = round(fraud_rate, 4)

    fraud_by_source: dict[str, float] = {}
    for src in df["data_source"].unique():
        sub  = df[df["data_source"] == src]
        rate = float(sub["fraud_label"].fillna(0).mean())
        fraud_by_source[src] = round(rate, 4)
    report["fraud_rate_by_source"] = fraud_by_source

    null_rates = (df.isna().mean() * 100).round(2)
    high_null  = null_rates[null_rates > 80].to_dict()
    report["high_null_columns"] = high_null
    if high_null:
        log.warning(
            "Columns >80%% null: %s",
            ", ".join(f"{k}={v:.1f}%%" for k, v in high_null.items()),
        )

    n_dupes = df.duplicated(subset=["person_id", "tax_year"]).sum()
    report["duplicate_person_year_rows"] = int(n_dupes)
    if n_dupes > 0:
        log.warning("Duplicate (person_id, tax_year) pairs: %d", n_dupes)

    if "agi" in df.columns:
        n_neg = int((df["agi"].dropna() < 0).sum())
        report["negative_agi_rows"] = n_neg
        if n_neg > 0:
            log.warning("Negative AGI rows: %d", n_neg)

    if "fraud_type" in df.columns:
        bad_clean = int(
            ((df["fraud_label"].fillna(0) == 0)
             & (df["fraud_type"] != "none")
             & df["fraud_type"].notna()).sum()
        )
        report["fraud_label_0_nonone_type"] = bad_clean
        if bad_clean > 0:
            log.warning("fraud_label=0 with fraud_type != none: %d", bad_clean)

    year_dist = df["tax_year"].value_counts().sort_index().to_dict()
    report["rows_by_year"] = {int(k): int(v) for k, v in year_dist.items()}

    zone_dist = df["zone"].value_counts().sort_index().to_dict()
    report["rows_by_zone"] = {str(k): int(v) for k, v in zone_dist.items()}

    return report


def _column_manifest(df: pd.DataFrame) -> list[dict]:
    manifest = []
    for col in df.columns:
        series  = df[col]
        n_null  = int(series.isna().sum())
        n_total = len(series)
        entry: dict = {
            "column":              col,
            "dtype":               str(series.dtype),
            "null_count":          n_null,
            "null_rate":           round(n_null / max(n_total, 1), 4),
            "in_canonical_schema": col in CANONICAL_COLUMNS,
        }
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                entry["min"]  = round(float(non_null.min()),  4)
                entry["max"]  = round(float(non_null.max()),  4)
                entry["mean"] = round(float(non_null.mean()), 4)
        else:
            entry["n_unique"] = int(series.nunique())
        manifest.append(entry)
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Main Modal function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=8,
    memory=65_536,
    timeout=7_200,
)
def merge_and_finalize():
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    from config import (
        TRAIN_ALL, VAL_2024, TEST_2025,
        GAN_W2, GAN_SE, GAN_ITEMIZERS,
        GAN_COMPLIANT_SEQ, GAN_EVADER_SEQ,
        MERGED_FULL, SCHEMA_DIR,
        FRAUD_RATE_OVERALL, FRAUD_RATE_TOLERANCE, RANDOM_SEED,
    )
    from utils import (
        get_logger, make_dirs,
        write_parquet, apply_null_policy,
        enforce_fraud_rate,
    )

    log = get_logger("10_merge", "10_merge.log")
    rng = np.random.default_rng(RANDOM_SEED + 10)

    make_dirs(MERGED_FULL.parent, SCHEMA_DIR)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    log.info("Step 1: Loading source files")

    def _safe_load(path: Path, label: str) -> Optional[pd.DataFrame]:
        if not path.exists():
            log.warning("MISSING: %s (%s) — skipping", label, path)
            return None
        df = pd.read_parquet(path)
        log.info("  %-25s  %8d rows  %d cols", label, len(df), len(df.columns))
        return df

    src_real_train = _safe_load(TRAIN_ALL,        "real_train")
    src_real_val   = _safe_load(VAL_2024,          "real_val_2024")
    src_real_test  = _safe_load(TEST_2025,         "real_test_2025")
    src_ctab_w2    = _safe_load(GAN_W2,            "ctab_w2")
    src_ctab_se    = _safe_load(GAN_SE,            "ctab_se")
    src_ctab_item  = _safe_load(GAN_ITEMIZERS,     "ctab_itemizers")
    src_tg_comp    = _safe_load(GAN_COMPLIANT_SEQ, "timegan_compliant")
    src_tg_evd     = _safe_load(GAN_EVADER_SEQ,    "timegan_evader")

    # ── Step 2: Align ─────────────────────────────────────────────────────────
    log.info("Step 2: Aligning to canonical schema (%d columns)",
             len(CANONICAL_COLUMNS))

    aligned: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}

    def _process(df: Optional[pd.DataFrame], tag: str) -> None:
        if df is None:
            return
        out = _align_to_schema(df, tag, rng)
        aligned.append(out)
        source_counts[tag] = len(out)
        log.info(
            "  %-25s  %8d rows  fraud=%.4f",
            tag, len(out),
            float(out["fraud_label"].fillna(0).mean()),
        )

    real_parts = [
        df for df in [src_real_train, src_real_val, src_real_test]
        if df is not None
    ]
    if real_parts:
        _process(pd.concat(real_parts, ignore_index=True), "real")

    _process(src_ctab_w2,   "ctab_w2")
    _process(src_ctab_se,   "ctab_se")
    _process(src_ctab_item, "ctab_itemizers")
    _process(src_tg_comp,   "timegan_compliant")
    _process(src_tg_evd,    "timegan_evader")

    if not aligned:
        raise RuntimeError("No source files loaded. Cannot produce output.")

    # ── Step 3: Concatenate ───────────────────────────────────────────────────
    log.info("Step 3: Concatenating %d sources", len(aligned))
    merged = pd.concat(aligned, ignore_index=True)
    log.info("Pre-dedup rows: %d", len(merged))

    # ── Step 4: Deduplicate ───────────────────────────────────────────────────
    log.info("Step 4: Deduplication")
    source_priority = {
        "real": 0, "ctab_w2": 1, "ctab_se": 2,
        "ctab_itemizers": 3, "timegan_compliant": 4, "timegan_evader": 5,
    }
    merged["_sort_key"] = merged["data_source"].map(source_priority).fillna(9)
    merged = (
        merged
        .sort_values("_sort_key")
        .drop_duplicates(subset=["person_id", "tax_year"], keep="first")
        .drop(columns=["_sort_key"])
        .reset_index(drop=True)
    )
    log.info("Post-dedup rows: %d", len(merged))

    # ── Step 5: Null policy ───────────────────────────────────────────────────
    log.info("Step 5: Applying null policy")
    merged = apply_null_policy(merged)

    # ── Step 6: Fraud consistency ─────────────────────────────────────────────
    log.info("Step 6: Fraud label consistency")
    clean_mask = merged["fraud_label"].fillna(0).astype(int) == 0
    for col in ("fraud_type", "fraud_category"):
        if col in merged.columns:
            merged.loc[clean_mask, col] = "none"
    for col in ("evasion_amount", "evasion_rate"):
        if col in merged.columns:
            merged.loc[clean_mask, col] = None

    # ── Step 7: Enforce fraud rate ────────────────────────────────────────────
    log.info("Step 7: Enforcing fraud rate %.4f ± %.4f",
             FRAUD_RATE_OVERALL, FRAUD_RATE_TOLERANCE)
    pre_rate = float(merged["fraud_label"].fillna(0).mean())
    log.info("  Pre-enforcement: %.4f", pre_rate)

    merged = enforce_fraud_rate(
        merged,
        target_rate  = FRAUD_RATE_OVERALL,
        tolerance    = FRAUD_RATE_TOLERANCE,
        label_col    = "fraud_label",
        random_state = RANDOM_SEED,
    )
    log.info("  Post-enforcement: %.4f",
             float(merged["fraud_label"].fillna(0).mean()))

    # ── Step 8: Shuffle ───────────────────────────────────────────────────────
    log.info("Step 8: Final shuffle")
    merged = (
        merged
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    # ── Step 9: Validate ──────────────────────────────────────────────────────
    log.info("Step 9: Validation")
    validation_report = _validate(merged, source_counts, log)
    log.info("  Total rows: %d", validation_report["total_rows"])
    log.info("  Fraud rate: %.4f", validation_report["fraud_rate_overall"])
    log.info("  By source:  %s", validation_report["source_breakdown"])
    log.info("  By year:    %s", validation_report["rows_by_year"])

    # ── Step 10: Write ────────────────────────────────────────────────────────
    log.info("Step 10: Writing outputs")

    write_parquet(merged, MERGED_FULL)
    log.info("  merged_full -> %s", MERGED_FULL)

    manifest_path = SCHEMA_DIR / "column_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(_column_manifest(merged), f, indent=2)

    report_path = SCHEMA_DIR / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    # Commit only volumes that were written to
    final_vol.commit()
    merged_vol.commit()
    logs_vol.commit()

    log.info("Merge complete. %d rows  fraud=%.4f",
             len(merged),
             float(merged["fraud_label"].fillna(0).mean()))

    return {
        "status":     "ok",
        "total_rows": len(merged),
        "fraud_rate": float(merged["fraud_label"].fillna(0).mean()),
        "sources":    source_counts,
    }


@app.local_entrypoint()
def main():
    result = merge_and_finalize.remote()
    print("\nMerge result:")
    for k, v in result.items():
        print(f"  {k}: {v}")