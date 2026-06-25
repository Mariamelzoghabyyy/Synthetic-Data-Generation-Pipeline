# 06_impossibility_flags.py — final calibrated version
# Changes vs previous:
#   - flag_low_withholding removed from total_flag_count (lift=1.11, fires 31.8%)
#   - flag_zero_tax_with_income threshold tightened: agi > 75,000 (was 50,000)
#   - flag_progressive_anomaly tightened: agi > 150,000, rate < 0.06 (was 100k, 0.08)
#   - flag_impossible_deductions kept as-is (mathematical significance)
#   - flag_benford_violation kept as-is (forensic significance)
#   - flag_hobby_loss_risk kept as-is (borderline GOOD)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

from config import ENGINEERED_DIR, YEARS, EPS

PHASE7_OWNED_FLAGS = [
    "flag_phantom_business",
    "flag_hobby_loss_risk",
    "flag_impossible_deductions",
    "flag_zero_tax_with_income",
    "flag_progressive_anomaly",
    "flag_high_bank_deposits",
    "flag_unexplained_wealth",
    "flag_benford_violation",
    "total_flag_count",
    "has_impossibility",
    "has_legal_violation",
    "has_lifestyle_anomaly",
    "has_forensic_violation",
]

PHASE1_OWNED_FLAGS = [
    "flag_irs_high",
    "flag_irs_very_high",
    "flag_irs_moderate",
    "flag_bank_deposit_extreme",
    "flag_bank_deposit_high",
    "flag_low_withholding",
]


def add_impossibility_flags(df: pd.DataFrame) -> pd.DataFrame:

    agi       = df["agi"].fillna(0)
    agi_abs   = agi.abs() + EPS
    net_se    = df["net_se_income"].fillna(0)       if "net_se_income"       in df.columns else pd.Series(0, index=df.index)
    gross_rec = df["gross_receipts"].fillna(0)      if "gross_receipts"      in df.columns else pd.Series(0, index=df.index)
    deductions= df["deduction_taken"].fillna(0)     if "deduction_taken"     in df.columns else pd.Series(0, index=df.index)
    total_tax = df["total_tax_liability"].fillna(0) if "total_tax_liability" in df.columns else pd.Series(0, index=df.index)
    eff_rate  = df["effective_tax_rate"].fillna(0)  if "effective_tax_rate"  in df.columns else pd.Series(0, index=df.index)
    bank_dep  = df["bank_deposit_ratio"].fillna(0)  if "bank_deposit_ratio"  in df.columns else pd.Series(1, index=df.index)

    # GROUP A: Legal violations
    df["flag_phantom_business"] = (
        (df["sch_c_total_expenses"].fillna(0) > 500) & (gross_rec == 0)
    ).astype("int8")

    df["flag_hobby_loss_risk"] = (
        (net_se < -5_000) & (gross_rec > 0)
    ).astype("int8")

    # GROUP B: Mathematical impossibilities
    df["flag_impossible_deductions"] = (
        (deductions > agi * 2) & (agi > 5_000)
    ).astype("int8")

    # GROUP C: Tax anomalies
    # Tightened: agi > 75,000 (was 50,000) to reduce legitimate EITC/credit cases
    df["flag_zero_tax_with_income"] = (
        (total_tax < 100) & (agi > 75_000)
    ).astype("int8")

    # Tightened: agi > 150,000 and rate < 0.06 (was 100k and 0.08)
    # Very high income with extremely low effective rate and few deductions
    df["flag_progressive_anomaly"] = (
        (agi > 150_000) &
        (eff_rate < 0.06) &
        (deductions < agi * 0.20)
    ).astype("int8")

    # GROUP D: Bank deposit anomaly
    df["flag_high_bank_deposits"] = (
        (bank_dep > 50.0) & (agi > 10_000)
    ).astype("int8")

    if "unexplained_wealth_ratio" in df.columns:
        df["flag_unexplained_wealth"] = (
            df["unexplained_wealth_ratio"].fillna(0) > 10.0
        ).astype("int8")
    else:
        df["flag_unexplained_wealth"] = np.int8(0)

    # GROUP E: Forensic
    if "benford_deviation_expenses" in df.columns:
        df["flag_benford_violation"] = (
            df["benford_deviation_expenses"].fillna(0) > 3.0
        ).astype("int8")
    else:
        df["flag_benford_violation"] = np.int8(0)

    # ── Summary rollups ───────────────────────────────────────────────────────
    # Excluded from total_flag_count:
    #   flag_irs_moderate   — backward signal
    #   flag_low_withholding — lift=1.11, fires 31.8%, adds noise to count
    #                          signal already captured by withholding_rate
    #                          and withholding_block_score as continuous features
    count_flag_cols = [
        c for c in df.columns
        if c.startswith("flag_")
        and not c.endswith("_block_score")
        and c not in ("flag_irs_moderate", "flag_low_withholding")
    ]
    df["total_flag_count"] = (
        df[count_flag_cols].sum(axis=1).clip(0, 127).astype("int8")
    )

    df["has_impossibility"] = (
        df[[
            "flag_impossible_deductions",
            "flag_zero_tax_with_income",
        ]].sum(axis=1) > 0
    ).astype("int8")

    df["has_legal_violation"] = (
        df[[
            "flag_phantom_business",
            "flag_hobby_loss_risk",
        ]].sum(axis=1) > 0
    ).astype("int8")

    df["has_lifestyle_anomaly"] = (
        df[[
            "flag_high_bank_deposits",
            "flag_unexplained_wealth",
        ]].sum(axis=1) > 0
    ).astype("int8")

    df["has_forensic_violation"] = (
        df["flag_benford_violation"].fillna(0)
    ).astype("int8")

    return df


def run():
    print("=" * 65)
    print("PHASE 7 -- IMPOSSIBILITY FLAGS (final calibration)")
    print("=" * 65)

    for year in YEARS:
        p = ENGINEERED_DIR / f"year_{year}_engineered.parquet"
        if not p.exists():
            print(f"  year_{year}: MISSING -- skipped")
            continue

        print(f"\n  year_{year}: loading...", end="")
        df = pd.read_parquet(p)

        drop_cols = [c for c in df.columns if c in PHASE7_OWNED_FLAGS]
        df = df.drop(columns=drop_cols, errors="ignore")
        n_before = df.shape[1]
        print(f" {len(df):,} rows")

        df = add_impossibility_flags(df)

        if year == YEARS[0]:
            all_flag_cols = [
                c for c in df.columns
                if c.startswith("flag_")
                and not c.endswith("_block_score")
            ]
            fraud_df = df[df["fraud_label"] == 1]
            clean_df = df[df["fraud_label"] == 0]

            print(f"\n    Flag calibration:")
            print(f"    {'Flag':<42} {'All%':>5}  {'Fraud%':>6}  {'Clean%':>6}  {'Lift':>5}  {'Owner':<7}  Status")
            print(f"    {'-' * 97}")

            for col in sorted(all_flag_cols):
                all_r = df[col].mean()
                fr_r  = fraud_df[col].mean()
                cl_r  = clean_df[col].mean()
                lift  = fr_r / (all_r + EPS)
                owner = "Phase1" if col in PHASE1_OWNED_FLAGS else "Phase7"

                if all_r > 0.60:
                    status = "[NOISE]"
                elif fr_r < cl_r:
                    status = "[BACKWARD]"
                elif lift > 1.3:
                    status = "[GOOD]"
                elif lift > 1.1:
                    status = "[OK]"
                else:
                    status = "[WEAK]"

                print(
                    f"    {col:<42} {all_r*100:>4.1f}%  "
                    f"{fr_r*100:>5.1f}%  {cl_r*100:>5.1f}%  "
                    f"{lift:>5.2f}  {owner:<7}  {status}"
                )

            fraud_mean = fraud_df["total_flag_count"].mean()
            clean_mean = clean_df["total_flag_count"].mean()
            sep        = fraud_mean / (clean_mean + EPS)
            print(
                f"\n    total_flag_count : "
                f"fraud={fraud_mean:.2f}  clean={clean_mean:.2f}  "
                f"ratio={sep:.2f}x"
            )

        print(f"\n    Flags added  : {df.shape[1] - n_before}")
        print(f"    Total cols   : {df.shape[1]}")
        df.to_parquet(p, index=False)
        print(f"    Saved        : {p.name}")
        del df; gc.collect()

    print(f"\n{'=' * 65}")
    print("PHASE 7 COMPLETE")


if __name__ == "__main__":
    run()