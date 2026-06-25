# 00_preprocessing.py
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — PREPROCESSING
#
# Key changes in this version:
#   - add_raw_signal_features() called BEFORE clip step
#   - After preprocessing, raw-only columns are merged in so they travel
#     through the entire pipeline. This means ml_ready files are built
#     from the engineered files directly — no late merge needed.
#   - bank_deposit_ratio now clips at 500 (was 20) per config
#   - federal_withheld added to SIGNAL_COLS imputation
#   - compute_expected_tax_rate fixed for inf bracket
#   - deduction_income_ratio derived if absent
# ═══════════════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc
import warnings
warnings.filterwarnings("ignore")

from config import (
    BY_YEAR, CLEAN_DIR, YEARS,
    INCOME_SOURCE_COLS, SCH_C_COLS, SIGNAL_COLS,
    ZERO_FILL_COLS, CLIP_BOUNDS, TAX_BRACKETS, EPS,
)


def compute_expected_tax_rate(agi: pd.Series) -> pd.Series:
    """
    Vectorized federal effective tax rate from AGI using progressive brackets.
    TAX_BRACKETS stores cumulative absolute thresholds.
    Final bracket uses float("inf") as sentinel — handled explicitly
    to prevent inf - inf = nan corrupting total_tax.
    All arithmetic in float64, returned as float32.
    """
    income    = agi.fillna(0.0).values.astype(np.float64)
    total_tax = np.zeros_like(income, dtype=np.float64)
    prev_top  = 0.0

    for bracket_top, rate in TAX_BRACKETS:
        if np.isfinite(bracket_top):
            bracket_width = bracket_top - prev_top
            in_bracket    = np.clip(income - prev_top, 0.0, bracket_width)
            prev_top      = bracket_top
        else:
            in_bracket = np.maximum(income - prev_top, 0.0)
        total_tax += in_bracket * rate

    expected_rate = np.where(
        income > 0.0,
        total_tax / np.maximum(income, EPS),
        0.0,
    )
    return pd.Series(expected_rate, index=agi.index, dtype=np.float32)


def safe_group_median(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    global_fallback: float,
) -> pd.Series:
    """
    GroupBy median with protection against all-null groups.
    np.nanmedian ignores NaN so partially-null groups use valid members.
    Falls back to global_fallback only when entire group is null.
    """
    def _nanmedian_fill(x: pd.Series) -> pd.Series:
        med      = np.nanmedian(x.values)
        fill_val = med if not np.isnan(med) else global_fallback
        return x.fillna(fill_val)

    return df.groupby(group_col)[col].transform(_nanmedian_fill)


# add_raw_signal_features() — replace this function in 00_preprocessing.py
# Recalibrated thresholds from actual data distributions (diagnose_flags.py output)

def add_raw_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from RAW column values BEFORE outlier clipping.

    Thresholds recalibrated from actual 2019 data distributions:

    bank_deposit_ratio:
      p25=1.34, p50=5.93, p75=32.30, p95=52.80, p99=4899, max=543,988
      fraud p50=8.08 vs clean p50=5.47
      Extreme values (p99=4899, max=543988) are the fraud signal.
      Old threshold >43 fired on 12% — reasonable but log transform
      captures the full range better than a hard threshold.
      New thresholds: >10 (p75 area), >100 (p99 area)

    irs_risk_score:
      p50=53.8, p75=95.7, p95=99.4
      fraud p50=86.9 vs clean p50=45.6 — massive separation
      Threshold >67 fires on 36.8% — reasonable
      Keep existing thresholds, they are correct.

    federal_withheld:
      fraud p50=85.8 vs clean p50=2458.9 — INVERTED, fraud has LOWER values
      flag_low_withholding: withheld < 1429 AND agi > 20000
      p25 of withheld = 0.0004 (many zeros), so threshold needs care.
      Use < 500 as the low withholding threshold (below fraud p50=85.8
      but captures the evasion zone).

    zone:
      fraud p50=2, clean p50=3 — lower zone = more fraud (confirmed)
      zone_risk = 6 - zone is correct encoding.
    """

    # ── bank_deposit_ratio ────────────────────────────────────────────────────
    # Raw range: -15.7 to 543,988. Negative values are data errors.
    # Signal is in the EXTREME right tail — log transform is critical.
    if "bank_deposit_ratio" in df.columns:
        bdr = df["bank_deposit_ratio"].fillna(0)

        # Log transform — handles the extreme right tail (max=543,988)
        # log1p(5.93)=1.89, log1p(52.8)=3.98, log1p(4899)=8.50, log1p(543988)=13.2
        df["bank_deposit_ratio_log"] = (
            np.log1p(np.maximum(bdr, 0))
        ).astype("float32")

        df["bank_deposit_ratio_sq_log"] = (
            df["bank_deposit_ratio_log"] ** 2
        ).astype("float32")

        # Recalibrated thresholds from actual data:
        # p75 = 32.3, p95 = 52.8, p99 = 4899
        # fraud p50 = 8.08, clean p50 = 5.47
        # >10  catches fraud p50 area — fires on ~35% (above p75 of clean)
        # >100 catches extreme cases — fires on ~14% above p99 area
        df["flag_bank_deposit_extreme"] = (bdr > 100.0).astype("int8")
        df["flag_bank_deposit_high"]    = (bdr > 10.0).astype("int8")

    # ── irs_risk_score ────────────────────────────────────────────────────────
    # Raw range: -1.34 to 103.96
    # fraud p50=86.9, clean p50=45.6 — largest single signal in dataset
    # Thresholds verified correct from actual data:
    #   >67  fires on 36.8% overall, much higher in fraud
    #   >97.5 fires on ~5% — very high risk tier
    if "irs_risk_score" in df.columns:
        irs = df["irs_risk_score"].fillna(0)

        df["flag_irs_high"]      = (irs > 67.0).astype("int8")
        df["flag_irs_very_high"] = (irs > 97.5).astype("int8")
        df["flag_irs_moderate"]  = (irs.between(51.0, 67.0)).astype("int8")
        df["irs_risk_score_sq"]  = ((irs ** 2) / 10_000.0).astype("float32")

    # ── federal_withheld ──────────────────────────────────────────────────────
    # Raw range: -623 to 138,262
    # fraud p50=85.8 vs clean p50=2458.9 — INVERTED signal
    # Fraud rows have MUCH LOWER withholding — tax evasion pattern
    # flag_low_withholding fires when withheld < 500 AND agi > 20,000
    # (captures the evasion zone around fraud p50 of 85.8)
    if "federal_withheld" in df.columns:
        fw = df["federal_withheld"].fillna(0)

        df["federal_withheld_log"] = (
            np.log1p(np.maximum(fw, 0))
        ).astype("float32")

        if "agi" in df.columns:
            agi_safe = df["agi"].fillna(0).abs() + EPS
            df["withholding_rate"] = (
                fw / agi_safe
            ).clip(-0.5, 1.0).astype("float32")

            # Recalibrated: fraud p50 withheld=85.8, so < 500 captures
            # the evasion zone. Old threshold was 1429 which was too high
            # (fired on clean taxpayers with moderate withholding).
            df["flag_low_withholding"] = (
                (fw < 500) & (df["agi"].fillna(0) > 20_000)
            ).astype("int8")
        else:
            df["withholding_rate"]     = np.float32(0.0)
            df["flag_low_withholding"] = np.int8(0)

    # ── zone ──────────────────────────────────────────────────────────────────
    # Verified: fraud p50=2, clean p50=3 — lower zone IS higher fraud risk
    # zone_risk = 6 - zone: zone=1 → risk=5, zone=5 → risk=1
    if "zone" in df.columns:
        zone = df["zone"].fillna(3).clip(1, 5)
        df["zone_risk"] = (6.0 - zone).astype("float32")

    return df


def merge_raw_columns(
    df_clean: pd.DataFrame,
    raw_path: Path,
    label: str,
) -> pd.DataFrame:
    """
    Merge raw-only columns into the cleaned/preprocessed dataframe
    on positional index (row order is preserved from preprocessing).

    Raw columns that were already processed (exist in df_clean) are
    skipped to avoid duplicates. Identifier columns are also skipped
    since they are already in df_clean or will be in ALWAYS_DROP.

    This ensures every downstream phase has access to raw columns
    (total_credits, gig_income, crypto_proceeds, etc.) for feature
    engineering without a late-stage merge.
    """
    if not raw_path.exists():
        print(f"    [WARN] Raw file not found: {raw_path.name} — skipping raw merge")
        return df_clean

    available_raw = set(pq.read_schema(raw_path).names)
    existing_cols = set(df_clean.columns)

    # Columns to pull from raw: present in raw but NOT yet in clean df
    # Skip identifiers that will be dropped later anyway
    skip_always = {"person_id"}
    cols_to_load = [
        c for c in available_raw
        if c not in existing_cols
        and c not in skip_always
    ]

    if not cols_to_load:
        print(f"    [{label}] No new raw columns to merge in")
        return df_clean

    raw_extra = pd.read_parquet(raw_path, columns=cols_to_load)

    # Row count guard
    if len(raw_extra) != len(df_clean):
        print(
            f"    [WARN] [{label}] Row mismatch raw={len(raw_extra):,} "
            f"clean={len(df_clean):,} — skipping raw merge"
        )
        return df_clean

    merged = pd.concat(
        [df_clean.reset_index(drop=True), raw_extra.reset_index(drop=True)],
        axis=1,
    )

    # Dedup guard (should never trigger but defensive)
    if merged.columns.duplicated().any():
        merged = merged.loc[:, ~merged.columns.duplicated()]

    print(
        f"    [{label}] raw merge: "
        f"+{len(cols_to_load)} cols → {merged.shape[1]} total"
    )
    return merged


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Guard categoricals against nulls before any groupby ──────────────────
    df["taxpayer_type"] = df["taxpayer_type"].fillna("UNKNOWN").astype(str)
    df["state"]         = df["state"].fillna("UNKNOWN").astype(str)

    # ── Fill zero-fill columns ────────────────────────────────────────────────
    for col in ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in INCOME_SOURCE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # ── Impute AGI ────────────────────────────────────────────────────────────
    if "agi" in df.columns:
        global_agi_median = float(np.nanmedian(df["agi"].dropna().values))
        income_sum = sum(
            df[c].fillna(0) for c in INCOME_SOURCE_COLS if c in df.columns
        )
        df["agi"] = df["agi"].fillna(income_sum.replace(0, np.nan))
        df["agi"] = safe_group_median(df, "agi", "taxpayer_type", global_agi_median)
        df["agi"] = df["agi"].fillna(global_agi_median)

    # ── RAW SIGNAL FEATURES — must come BEFORE clip step ─────────────────────
    df = add_raw_signal_features(df)

    # ── Gross reconstruction ──────────────────────────────────────────────────
    income_cols_present = [c for c in INCOME_SOURCE_COLS if c in df.columns]
    df["gross_reconstructed_income_sum"] = (
        df[income_cols_present].fillna(0).sum(axis=1)
    ).astype("float32")

    df["gross_income_mismatch"] = (
        df["gross_reconstructed_income_sum"] - df["agi"]
    ).clip(-1_000_000, 1_000_000).astype("float32")

    df["gross_income_mismatch_ratio"] = (
        df["gross_income_mismatch"].abs() / (df["agi"].abs() + EPS)
    ).clip(0, 5).astype("float32")

    # ── Derive deduction_income_ratio if absent ───────────────────────────────
    if "deduction_income_ratio" not in df.columns:
        if "deduction_taken" in df.columns:
            df["deduction_income_ratio"] = (
                df["deduction_taken"].fillna(0) / (df["agi"].abs() + EPS)
            ).clip(0, 5).astype("float32")
        else:
            df["deduction_income_ratio"] = np.float32(0.0)

    # ── Impute effective_tax_rate ─────────────────────────────────────────────
    if "effective_tax_rate" in df.columns and "total_tax_liability" in df.columns:
        computed_rate = (
            df["total_tax_liability"].fillna(0) / (df["agi"].abs() + EPS)
        ).clip(0, 0.65)
        df["effective_tax_rate"] = df["effective_tax_rate"].fillna(computed_rate)

    # ── Impute age ────────────────────────────────────────────────────────────
    if "age" in df.columns:
        df["age"] = safe_group_median(df, "age", "taxpayer_type", 45.0)
        df["age"] = df["age"].fillna(45.0)

    # ── Impute signal columns ─────────────────────────────────────────────────
    for col in SIGNAL_COLS:
        if col in df.columns:
            global_med = float(np.nanmedian(df[col].dropna().values))
            df[col]    = safe_group_median(df, col, "taxpayer_type", global_med)
            df[col]    = df[col].fillna(global_med)

    # ── Clip outliers ─────────────────────────────────────────────────────────
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # ── Fix structural impossibilities ────────────────────────────────────────
    if "taxable_income" in df.columns and "agi" in df.columns:
        df["taxable_income"] = np.minimum(
            df["taxable_income"].fillna(0), df["agi"]
        )
    if "total_tax_liability" in df.columns and "agi" in df.columns:
        df["total_tax_liability"] = np.minimum(
            df["total_tax_liability"].fillna(0), df["agi"]
        )
    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=16, upper=100)

    # ── Add expected tax rate ─────────────────────────────────────────────────
    if "agi" in df.columns:
        df["expected_tax_rate"] = compute_expected_tax_rate(df["agi"])

    # ── Downcast dtypes ───────────────────────────────────────────────────────
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def run():
    print("=" * 65)
    print("PHASE 1 -- PREPROCESSING")
    print("=" * 65)
    print(f"\n  BY_YEAR path    : {BY_YEAR}")
    print(f"  BY_YEAR exists  : {BY_YEAR.exists()}")
    print(f"  CLEAN_DIR       : {CLEAN_DIR}")

    if not BY_YEAR.exists():
        print("  [FAIL] BY_YEAR folder not found -- check config.py DATA_ROOT")
        return

    total_rows         = 0
    total_nulls_before = 0
    total_nulls_after  = 0

    for year in YEARS:
        p_in  = BY_YEAR   / f"year_{year}.parquet"
        p_out = CLEAN_DIR / f"year_{year}_clean.parquet"

        if not p_in.exists():
            print(f"  year_{year}: MISSING -- skipped")
            continue

        print(f"\n  year_{year}:")
        df           = pd.read_parquet(p_in)
        nulls_before = int(df.isnull().sum().sum())
        total_rows  += len(df)
        total_nulls_before += nulls_before
        print(f"    Loaded       : {len(df):,} rows x {df.shape[1]} cols")
        print(f"    Nulls before : {nulls_before:,}")

        df_clean = preprocess(df)
        del df; gc.collect()

        # ── Merge raw-only columns into clean file ────────────────────────────
        # This is the key change: raw columns (gig_income, crypto_proceeds,
        # total_credits, etc.) are merged in NOW so every downstream phase
        # can engineer features from them without a separate late-stage merge.
        df_clean = merge_raw_columns(df_clean, p_in, f"year_{year}")

        nulls_after        = int(df_clean.isnull().sum().sum())
        total_nulls_after += nulls_after
        print(f"    Nulls after  : {nulls_after:,}")
        print(f"    Cols after   : {df_clean.shape[1]}")

        if year == YEARS[0]:
            raw_feats = [
                c for c in df_clean.columns
                if c in (
                    "bank_deposit_ratio_log", "bank_deposit_ratio_sq_log",
                    "flag_bank_deposit_extreme", "flag_bank_deposit_high",
                    "flag_irs_high", "flag_irs_very_high", "flag_irs_moderate",
                    "irs_risk_score_sq", "withholding_rate",
                    "flag_low_withholding", "federal_withheld_log", "zone_risk",
                )
            ]
            print(f"    Raw signal features : {len(raw_feats)}")

        df_clean.to_parquet(p_out, index=False)
        print(f"    Saved        : {p_out.name}")
        del df_clean; gc.collect()

    print(f"\n{'=' * 65}")
    print("PHASE 1 COMPLETE")
    print(f"  Total rows      : {total_rows:,}")
    print(f"  Nulls before    : {total_nulls_before:,}")
    print(f"  Nulls after     : {total_nulls_after:,}")
    if total_nulls_before > 0:
        pct = (1 - total_nulls_after / total_nulls_before) * 100
        print(f"  Null reduction  : {pct:.1f}%")
    print(f"  Output          : {CLEAN_DIR}")


if __name__ == "__main__":
    run()