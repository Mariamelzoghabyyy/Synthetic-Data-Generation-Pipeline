# quality_checks.py
"""
Per-stage validation for the full synthetic tax pipeline.

Usage
-----
  python quality_checks.py            # run all checks
  python quality_checks.py --stage 1  # run only stage 1 check
  python quality_checks.py --stage 5 --year 2021
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DIST_PKL, DIST_JSON,
    PERSONS_CSV, BUSINESSES_CSV, PB_LINKS_CSV, EMP_LINKS_CSV,
    PANEL_BY_DIR, PANEL_BZY_DIR,
    TRAIN_ALL, TRAIN_EMPLOYEES, TRAIN_SE, TRAIN_ITEMIZERS,
    VAL_2024, TEST_2025,
    SEED_W2, SEED_SE, SEED_ITEMIZERS, SEED_COMPLIANT, SEED_EVADERS,
    GAN_W2, GAN_SE, GAN_ITEMIZERS,
    GAN_COMPLIANT_SEQ, GAN_EVADER_SEQ,
    MERGED_FULL, MERGED_DIR,
    FRAUD_RATE_OVERALL, FRAUD_RATE_W2, FRAUD_RATE_SE,
    FRAUD_RATE_ITEMIZERS, FRAUD_RATE_TOLERANCE,
    YEARS, ZONES,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
SEP  = "=" * 62

_failures: list = []


def _assert(condition: bool, message: str) -> None:
    if not condition:
        print(f"  {FAIL}  {message}")
        _failures.append(message)
    else:
        print(f"  {PASS}  {message}")


def _warn(condition: bool, message: str) -> None:
    if not condition:
        print(f"  {WARN}  {message}")
    else:
        print(f"  {PASS}  {message}")


# ── Stage 1: Distributions ────────────────────────────────────────────────────

def check_stage_1():
    """Run after: convert_distributions.py or 01_extract_distributions.py"""
    print(f"\n{SEP}\nSTAGE 1 -- Distributions\n{SEP}")
    import pickle

    _assert(DIST_PKL.exists(), f"master_distributions.pkl exists at {DIST_PKL}")
    if not DIST_PKL.exists():
        print(f"  -> Run convert_distributions.py first")
        return

    with open(DIST_PKL, "rb") as f:
        dist = pickle.load(f)

    required_keys = ["acs", "bls", "schedule_c",
                     "housing", "zone_profiles", "macro_shocks"]
    for key in required_keys:
        _assert(key in dist, f"dist contains key '{key}'")

    _assert(len(dist["zone_profiles"]) == 5,
            "zone_profiles has exactly 5 zones")
    _assert(len(dist["macro_shocks"]) == len(YEARS),
            f"macro_shocks has {len(YEARS)} years")
    _assert(len(dist["bls"]["occupations"]) >= 10,
            "BLS has >= 10 occupations")
    _assert(len(dist["schedule_c"]) >= 5,
            "Schedule-C has >= 5 industries")

    for zone in range(1, 6):
        zp = dist["zone_profiles"][zone]
        _assert("income_multiplier" in zp,
                f"zone {zone} has income_multiplier")
        _assert("fraud_base_rate" in zp,
                f"zone {zone} has fraud_base_rate")
        _assert(zone in dist["housing"],
                f"housing data exists for zone {zone}")

    # Print summary
    print(f"\n  Zones:       {list(dist['zone_profiles'].keys())}")
    print(f"  Years:       {list(dist['macro_shocks'].keys())}")
    print(f"  Occupations: {len(dist['bls']['occupations'])}")
    print(f"  Industries:  {len(dist['schedule_c'])}")
    print(f"  ACS wage p50: ${dist['acs']['wage_income'].get('p50', 0):,.0f}")
    for zone in range(1, 6):
        hv   = dist["housing"][zone]["median_home_value"]
        mult = dist["zone_profiles"][zone]["income_multiplier"]
        print(f"  Zone {zone}: home=${hv:>9,.0f}  "
              f"income_mult={mult:.3f}")


# ── Stage 2: Persons ──────────────────────────────────────────────────────────

def check_stage_2():
    """Run after: 02_generate_persons.py"""
    print(f"\n{SEP}\nSTAGE 2 -- Persons\n{SEP}")
    _assert(PERSONS_CSV.exists(),
            f"persons.csv exists at {PERSONS_CSV}")
    if not PERSONS_CSV.exists():
        print(f"  -> Run 02_generate_persons.py first")
        return

    df = pd.read_csv(PERSONS_CSV)
    print(f"  Rows: {len(df):,}")

    _assert(len(df) > 400_000, "Total persons > 400,000")
    _assert(df["person_id"].nunique() == len(df),
            "No duplicate person_ids")
    _assert(df["age_at_entry"].between(18, 80).all(),
            "All ages in [18, 80]")
    _assert(df["taxpayer_type"].notna().all(),
            "No null taxpayer_type")
    _assert(df["fraud_persona"].notna().all(),
            "No null fraud_persona")
    _assert(df["risk_score_base"].between(1, 99).all(),
            "risk_score_base in [1, 99]")
    _assert(df["zone"].isin([1, 2, 3, 4, 5]).all(),
            "All zones in {1,2,3,4,5}")

    clean_rate = (df["fraud_persona"] == "clean").mean()
    _assert(0.75 < clean_rate < 0.92,
            f"Clean persona rate {clean_rate:.3f} in (0.75, 0.92)")

    # Critical leakage check
    df["_is_evader"] = (df["fraud_persona"] != "clean").astype(int)
    corr = df["risk_score_base"].corr(df["_is_evader"])
    print(f"\n  risk_score_base ~ evader-persona correlation: {corr:.4f}")
    _assert(corr < 0.25,
            f"risk_score_base does not leak fraud_persona "
            f"(corr={corr:.4f} < 0.25)")

    print(f"\n  Taxpayer type distribution:")
    print(df["taxpayer_type"].value_counts(
        normalize=True).round(3).to_string())
    print(f"\n  Fraud persona distribution:")
    print(df["fraud_persona"].value_counts(
        normalize=True).round(4).to_string())


# ── Stage 3: Businesses ───────────────────────────────────────────────────────

def check_stage_3():
    """Run after: 03_generate_businesses.py"""
    print(f"\n{SEP}\nSTAGE 3 -- Businesses\n{SEP}")
    _assert(BUSINESSES_CSV.exists(), "businesses.csv exists")
    if not BUSINESSES_CSV.exists():
        print(f"  -> Run 03_generate_businesses.py first")
        return

    df = pd.read_csv(BUSINESSES_CSV)
    print(f"  Rows: {len(df):,}")

    _assert(len(df) > 100_000, "Total businesses > 100,000")
    _assert(df["business_id"].nunique() == len(df),
            "No duplicate business_ids")
    _assert(df["zone"].isin([1, 2, 3, 4, 5]).all(),
            "All zones valid")
    _assert(df["industry"].notna().all(), "No null industry")
    _assert(df["entity_type"].notna().all(), "No null entity_type")
    _assert((df["n_employees_base"] >= 0).all(),
            "n_employees_base non-negative")

    has_exit = df["exit_year"].notna().mean()
    _warn(has_exit > 0.30,
          f"Exit year set for {has_exit:.1%} of businesses")

    print(f"\n  Entity type distribution:")
    print(df["entity_type"].value_counts(
        normalize=True).round(3).to_string())
    print(f"\n  Industry distribution:")
    print(df["industry"].value_counts(
        normalize=True).round(3).to_string())


# ── Stage 4: Links ────────────────────────────────────────────────────────────

def check_stage_4():
    """Run after: 04_generate_links.py"""
    print(f"\n{SEP}\nSTAGE 4 -- Links\n{SEP}")
    _assert(PB_LINKS_CSV.exists(),
            "person_business_links.csv exists")
    _assert(EMP_LINKS_CSV.exists(),
            "employment_links.csv exists")

    if PB_LINKS_CSV.exists():
        pb = pd.read_csv(PB_LINKS_CSV)
        print(f"  Ownership links: {len(pb):,}")
        _assert(len(pb) > 10_000, "Ownership links > 10,000")
        _assert(pb["person_id"].notna().all(),
                "No null person_id in PB links")
        _assert(pb["business_id"].notna().all(),
                "No null business_id in PB links")
        _assert((pb["ownership_pct"] > 0).all(),
                "All ownership_pct > 0")
        _assert(
            (pb["ownership_start"] <=
             pb["ownership_end"].fillna(2025)).all(),
            "ownership_start <= ownership_end"
        )

    if EMP_LINKS_CSV.exists():
        emp = pd.read_csv(EMP_LINKS_CSV)
        print(f"  Employment links: {len(emp):,}")
        _assert(len(emp) > 50_000, "Employment links > 50,000")
        _assert(emp["person_id"].notna().all(),
                "No null person_id in emp links")
        _assert((emp["base_salary_2019"] > 0).all(),
                "All base salaries > 0")
        _assert(
            (emp["employment_start"] <=
             emp["employment_end"].fillna(2025)).all(),
            "employment_start <= employment_end"
        )
        _assert(emp["is_part_time"].isin([0, 1]).all(),
                "is_part_time is binary")


# ── Stage 5: Panels ───────────────────────────────────────────────────────────

def check_stage_5(years=None):
    """Run after: 05_generate_panels.py"""
    print(f"\n{SEP}\nSTAGE 5 -- Panels\n{SEP}")
    years = years or YEARS
    fraud_rates = {}

    for year in years:
        path = PANEL_BY_DIR / f"all_zones_{year}.parquet"
        _assert(path.exists(), f"Panel file exists for {year}")
        if not path.exists():
            continue

        df         = pd.read_parquet(path)
        fraud_rate = df["fraud_label"].mean()
        fraud_rates[year] = fraud_rate
        print(f"\n  Year {year}: {len(df):,} rows  "
              f"fraud={fraud_rate:.4f}")

        for col in ["person_id", "year", "zone", "fraud_label",
                    "agi", "tax_liability", "irs_risk_score"]:
            n_null = df[col].isna().sum()
            _assert(n_null == 0,
                    f"{year}/{col}: no nulls ({n_null} found)")

        _assert((df["agi"] >= 0).all(),
                f"{year}: AGI non-negative")
        _assert((df["taxable_income"] <= df["agi"] + 1).all(),
                f"{year}: taxable_income <= AGI")
        _assert((df["tax_liability"] >= 0).all(),
                f"{year}: tax_liability non-negative")
        _assert(df["effective_tax_rate"].between(0, 0.37).all(),
                f"{year}: effective_tax_rate in [0, 0.37]")
        _assert(df["irs_risk_score"].between(1, 99).all(),
                f"{year}: irs_risk_score in [1, 99]")
        _assert(0.05 < fraud_rate < 0.50,
                f"{year}: fraud rate {fraud_rate:.4f} in (0.05, 0.50)")

        for sig in ["sig_lifestyle_gap", "sig_cash_intensity",
                    "sig_unreported_income", "sig_inflated_deductions"]:
            if sig in df.columns:
                _assert(df[sig].between(0, 1).all(),
                        f"{year}/{sig} in [0, 1]")

        non_se = df[~df["taxpayer_type"].isin([
            "pure_se", "w2_with_side_biz", "business_owner",
            "multi_biz_owner", "gig_only"
        ])]
        if "gross_receipts" in df.columns:
            bad = non_se["gross_receipts"].notna().sum()
            _assert(bad == 0,
                    f"{year}: non-SE rows with gross_receipts ({bad})")

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(
                df["fraud_label"], df["irs_risk_score"]
            )
            print(f"    irs_risk_score AUC: {auc:.4f}")
            _assert(auc < 0.80,
                    f"{year}: IRS risk AUC {auc:.4f} < 0.80")
        except ImportError:
            print(f"    (sklearn not available - skipping AUC check)")

        for zone in ZONES:
            zpath = PANEL_BZY_DIR / f"zone_{zone}_{year}.parquet"
            _assert(zpath.exists(),
                    f"zone_{zone}_{year}.parquet exists")

    if len(fraud_rates) > 1:
        rates = list(fraud_rates.values())
        drift = max(rates) - min(rates)
        _warn(drift < 0.15,
              f"Fraud rate drift across years: {drift:.4f} (want < 0.15)")


# ── Stage 6: Splits ───────────────────────────────────────────────────────────

def check_stage_6():
    """Run after: 06_splits.py"""
    print(f"\n{SEP}\nSTAGE 6 -- Train / Val / Test Splits\n{SEP}")

    for path, label in [
        (TRAIN_ALL,       "train_all"),
        (TRAIN_EMPLOYEES, "train_employees"),
        (TRAIN_SE,        "train_se"),
        (VAL_2024,        "val_2024"),
        (TEST_2025,       "test_2025"),
    ]:
        _assert(path.exists(), f"{label} exists")

    if not TRAIN_ALL.exists():
        print("  -> Run 06_splits.py first")
        return

    train = pd.read_parquet(TRAIN_ALL)
    val   = pd.read_parquet(VAL_2024)
    test  = pd.read_parquet(TEST_2025)

    print(f"\n  train_all: {len(train):,}  "
          f"fraud={train['fraud_label'].mean():.4f}")
    print(f"  val_2024:  {len(val):,}  "
          f"fraud={val['fraud_label'].mean():.4f}")
    print(f"  test_2025: {len(test):,}  "
          f"fraud={test['fraud_label'].mean():.4f}")

    train_rate = train["fraud_label"].mean()
    _assert(
        abs(train_rate - FRAUD_RATE_OVERALL) <=
        FRAUD_RATE_TOLERANCE + 0.01,
        f"Train fraud rate {train_rate:.4f} ~= {FRAUD_RATE_OVERALL}"
    )

    _warn(0.05 < val["fraud_label"].mean() < 0.50,
          f"Val fraud rate plausible: {val['fraud_label'].mean():.4f}")
    _warn(0.05 < test["fraud_label"].mean() < 0.50,
          f"Test fraud rate plausible: {test['fraud_label'].mean():.4f}")

    if "year" in test.columns:
        _assert((test["year"] == 2025).all(),
                "test_2025 contains only year=2025")
    if "year" in val.columns:
        _assert((val["year"] == 2024).all(),
                "val_2024 contains only year=2024")

    if TRAIN_EMPLOYEES.exists():
        emp = pd.read_parquet(TRAIN_EMPLOYEES)
        emp_rate = emp["fraud_label"].mean()
        _assert(
            abs(emp_rate - FRAUD_RATE_W2) <= FRAUD_RATE_TOLERANCE + 0.01,
            f"Employee fraud rate {emp_rate:.4f} ~= {FRAUD_RATE_W2}"
        )

    if TRAIN_SE.exists():
        se = pd.read_parquet(TRAIN_SE)
        se_rate = se["fraud_label"].mean()
        _assert(
            abs(se_rate - FRAUD_RATE_SE) <= FRAUD_RATE_TOLERANCE + 0.01,
            f"SE fraud rate {se_rate:.4f} ~= {FRAUD_RATE_SE}"
        )


# ── Stage 7: Seed split ───────────────────────────────────────────────────────

def check_stage_7():
    """Run after: 07_seed_panel_splitter.py"""
    print(f"\n{SEP}\nSTAGE 7 -- Seed Panel Split\n{SEP}")

    for path, label in [
        (SEED_W2,        "seed_w2"),
        (SEED_SE,        "seed_se"),
        (SEED_COMPLIANT, "seed_compliant"),
        (SEED_EVADERS,   "seed_evaders"),
    ]:
        _assert(path.exists(), f"{label} exists")
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  {label}: {len(df):,} rows  "
                  f"fraud={df['fraud_label'].mean():.4f}")

    if SEED_COMPLIANT.exists():
        comp = pd.read_parquet(SEED_COMPLIANT)
        _assert(comp["fraud_label"].sum() == 0,
                "seed_compliant has zero fraud rows")

    if SEED_EVADERS.exists():
        evad = pd.read_parquet(SEED_EVADERS)
        _assert(evad["fraud_label"].mean() == 1.0,
                "seed_evaders has 100% fraud rows")


# ── Stage 8: CTAB-GAN ─────────────────────────────────────────────────────────

def check_stage_8():
    """Run after: 08_ctab_gan_runner.py"""
    print(f"\n{SEP}\nSTAGE 8 -- CTAB-GAN Outputs\n{SEP}")

    configs = [
        (GAN_W2,        "ctabgan_w2",        FRAUD_RATE_W2),
        (GAN_SE,        "ctabgan_se",        FRAUD_RATE_SE),
        (GAN_ITEMIZERS, "ctabgan_itemizers", FRAUD_RATE_ITEMIZERS),
    ]

    for path, label, target_rate in configs:
        if not path.exists():
            print(f"  {WARN}  {label}: not found - skipping")
            continue
        df = pd.read_parquet(path)
        fraud_rate = df["fraud_label"].mean()
        print(f"\n  {label}: {len(df):,} rows  fraud={fraud_rate:.4f}")

        _assert(len(df) > 100_000,
                f"{label}: row count > 100,000")
        _assert(
            abs(fraud_rate - target_rate) <= FRAUD_RATE_TOLERANCE + 0.02,
            f"{label}: fraud rate {fraud_rate:.4f} ~= {target_rate}"
        )

        for col in ["agi", "w2_wages", "gross_receipts"]:
            if col in df.columns:
                n_neg = (df[col].fillna(0) < 0).sum()
                _assert(n_neg == 0,
                        f"{label}/{col}: no negative values")

        if "effective_tax_rate" in df.columns:
            _assert(df["effective_tax_rate"].between(0, 0.37).all(),
                    f"{label}/effective_tax_rate in [0, 0.37]")

        if "fraud_type" in df.columns:
            bad = ((df["fraud_label"] == 0) &
                   (df["fraud_type"] != "none")).sum()
            _assert(bad == 0,
                    f"{label}: fraud_type='none' for clean records")


# ── Stage 9: TimeGAN ──────────────────────────────────────────────────────────

def check_stage_9():
    """Run after: 09_timegan_runner.py"""
    print(f"\n{SEP}\nSTAGE 9 -- TimeGAN Outputs\n{SEP}")

    for path, label, expect_fraud in [
        (GAN_COMPLIANT_SEQ, "timegan_compliant", False),
        (GAN_EVADER_SEQ,    "timegan_evaders",   True),
    ]:
        if not path.exists():
            print(f"  {WARN}  {label}: not found - skipping")
            continue
        df = pd.read_parquet(path)
        fraud_rate = df["fraud_label"].mean()
        print(f"\n  {label}: {len(df):,} rows  fraud={fraud_rate:.4f}")

        _assert(len(df) > 10_000,
                f"{label}: row count > 10,000")

        if expect_fraud:
            _assert(fraud_rate > 0.30,
                    f"{label}: fraud rate > 0.30 (evader model)")
        else:
            _assert(fraud_rate < 0.15,
                    f"{label}: fraud rate < 0.15 (compliant model)")

        if "person_id" in df.columns and "year" in df.columns:
            ypp    = df.groupby("person_id")["year"].nunique()
            median = ypp.median()
            print(f"    Median years per person: {median:.1f}")
            _assert(median >= 3,
                    f"{label}: median years per person >= 3")

        for col in ["fraud_label", "agi"]:
            if col in df.columns:
                n_null = df[col].isna().sum()
                _assert(n_null == 0, f"{label}/{col}: no nulls")


# ── Stage 10: Merger ──────────────────────────────────────────────────────────

def check_stage_10():
    """Run after: 10_gan_merger.py"""
    print(f"\n{SEP}\nSTAGE 10 -- Merged Output\n{SEP}")

    _assert(MERGED_FULL.exists(),
            f"merged_full.parquet exists at {MERGED_FULL}")
    if not MERGED_FULL.exists():
        print("  -> Run 10_gan_merger.py first")
        return

    df = pd.read_parquet(MERGED_FULL)
    print(f"  Total rows:     {len(df):,}")
    print(f"  Unique persons: {df['person_id'].nunique():,}")
    print(f"  Fraud rate:     {df['fraud_label'].mean():.4f}")

    _assert(len(df) > 1_000_000,
            "merged_full has > 1M rows")
    _assert(df["person_id"].nunique() > 100_000,
            "merged_full has > 100K unique persons")
    _assert("data_source" in df.columns,
            "data_source column present")

    if "data_source" in df.columns:
        print(f"\n  Source breakdown:")
        for src, grp in df.groupby("data_source"):
            print(f"    {src:<22} {len(grp):>10,} rows  "
                  f"fraud={grp['fraud_label'].mean():.4f}")
        sources = set(df["data_source"].unique())
        _assert("seed_panel" in sources,
                "seed_panel source present")
        _assert("ctabgan" in sources,
                "ctabgan source present")

    for col in ["person_id", "fraud_label", "fraud_type",
                "agi", "taxable_income", "tax_liability",
                "effective_tax_rate", "irs_risk_score"]:
        _assert(col in df.columns,
                f"Column '{col}' present in merged_full")

    _assert((df["agi"] >= 0).all(),
            "All AGI values non-negative")
    _assert((df["taxable_income"] <= df["agi"] + 1).all(),
            "taxable_income <= AGI everywhere")
    _assert(df["effective_tax_rate"].between(0, 0.37).all(),
            "effective_tax_rate in [0, 0.37]")
    _assert(df["irs_risk_score"].between(1, 99).all(),
            "irs_risk_score in [1, 99]")

    clean_mask = df["fraud_label"] == 0
    if "fraud_type" in df.columns:
        bad = (clean_mask & (df["fraud_type"] != "none")).sum()
        _assert(bad == 0,
                f"fraud_type='none' for all clean rows ({bad} violations)")

    for sig in ["sig_lifestyle_gap", "sig_cash_intensity",
                "sig_unreported_income", "sig_inflated_deductions"]:
        if sig in df.columns:
            _assert(df[sig].between(0, 1).all(),
                    f"{sig} in [0, 1]")

    if "data_source" in df.columns:
        seed_ids = set(df.loc[
            df["data_source"] == "seed_panel", "person_id"
        ].unique())
        gan_ids = set(df.loc[
            df["data_source"] != "seed_panel", "person_id"
        ].unique())
        overlap = seed_ids & gan_ids
        _assert(len(overlap) == 0,
                f"No person_id collisions between seed and GAN "
                f"({len(overlap)} found)")

    print(f"\n  Checking rebuilt splits...")
    if TRAIN_ALL.exists():
        train      = pd.read_parquet(TRAIN_ALL)
        train_rate = train["fraud_label"].mean()
        _assert(
            abs(train_rate - FRAUD_RATE_OVERALL) <=
            FRAUD_RATE_TOLERANCE + 0.01,
            f"Rebuilt train fraud rate {train_rate:.4f} ~= "
            f"{FRAUD_RATE_OVERALL}"
        )

    if VAL_2024.exists():
        val = pd.read_parquet(VAL_2024)
        _assert(
            "year" not in val.columns or
            (val["year"] == 2024).all(),
            "Rebuilt val_2024 contains only year=2024"
        )

    if TEST_2025.exists():
        test = pd.read_parquet(TEST_2025)
        _assert(
            "year" not in test.columns or
            (test["year"] == 2025).all(),
            "Rebuilt test_2025 contains only year=2025"
        )
        _assert(
            "data_source" not in test.columns or
            (test["data_source"] == "seed_panel").all(),
            "Rebuilt test_2025 is seed-only (no GAN rows)"
        )

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(df["fraud_label"], df["irs_risk_score"])
        print(f"\n  irs_risk_score AUC vs fraud_label: {auc:.4f}")
        _assert(auc < 0.80,
                f"irs_risk_score AUC {auc:.4f} < 0.80 (no leakage)")
    except ImportError:
        print("  (sklearn not available - skipping AUC leakage test)")


# ── Stage map ─────────────────────────────────────────────────────────────────

STAGE_MAP = {
    1:  check_stage_1,
    2:  check_stage_2,
    3:  check_stage_3,
    4:  check_stage_4,
    6:  check_stage_6,
    7:  check_stage_7,
    8:  check_stage_8,
    9:  check_stage_9,
    10: check_stage_10,
}


def run_all():
    check_stage_1()
    check_stage_2()
    check_stage_3()
    check_stage_4()
    check_stage_5()
    check_stage_6()
    check_stage_7()
    check_stage_8()
    check_stage_9()
    check_stage_10()

    print(f"\n{SEP}")
    if _failures:
        print(f"\033[91m{len(_failures)} CHECK(S) FAILED:\033[0m")
        for f in _failures:
            print(f"  x  {f}")
        sys.exit(1)
    else:
        print(f"\033[92mALL CHECKS PASSED\033[0m")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run quality checks for the synthetic tax pipeline."
    )
    parser.add_argument(
        "--stage", type=int, default=None,
        help="Stage to check (1-10). Omit to run all.",
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Specific year for stage 5 only.",
    )
    args = parser.parse_args()

    if args.stage is None:
        run_all()
    elif args.stage == 5:
        years = [args.year] if args.year else None
        check_stage_5(years)
        if _failures:
            sys.exit(1)
    elif args.stage in STAGE_MAP:
        STAGE_MAP[args.stage]()
        if _failures:
            sys.exit(1)
    else:
        print(f"Unknown stage {args.stage}. Valid: 1-10.")
        sys.exit(1)