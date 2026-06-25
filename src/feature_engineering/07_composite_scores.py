# 07_composite_scores.py
# Fix 5 applied: flag_low_wh uses safe fallback chain
# Fix 6 applied: dead constant block scores removed entirely
# ═══════════════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

from config import (
    ENGINEERED_DIR, YEARS,
    VALID_STATES, VALID_TAXPAYER_TYPES, VALID_AGE_GROUPS,
    FIXED_NORM_BOUNDS, EPS,
)


def norm01(series: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi <= lo:
        return pd.Series(0.0, index=series.index, dtype=np.float32)
    return (
        (series - lo) / (hi - lo + EPS)
    ).clip(0.0, 1.0).astype("float32")


def stable_norm01(series: pd.Series, name: str) -> pd.Series:
    if name.startswith(("flag_", "has_")) or series.nunique() <= 2:
        return series.fillna(0.0).astype("float32")
    if name in FIXED_NORM_BOUNDS:
        lo, hi = FIXED_NORM_BOUNDS[name]
        return norm01(series.fillna(0), lo, hi)
    lo = float(series.quantile(0.01))
    hi = float(series.quantile(0.99))
    return norm01(series.fillna(0), lo, hi)


def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:

    # BLOCK 1 — IRS RISK SCORE
    if "irs_risk_score" in df.columns:
        irs = df["irs_risk_score"].fillna(0).clip(0, 100)
        irs_nl = np.where(
            irs.values <= 60.0,
            irs.values / 60.0 * 0.30,
            0.30 + (irs.values - 60.0) / 40.0 * 0.70,
        )
        df["irs_block_score"]    = norm01(irs, 0.0, 100.0)
        df["irs_block_score_nl"] = np.clip(irs_nl, 0.0, 1.0).astype("float32")
    else:
        df["irs_block_score"]    = np.float32(0.0)
        df["irs_block_score_nl"] = np.float32(0.0)

# In add_composite_scores() — replace BLOCK 2

    # BLOCK 2 — BANK DEPOSIT SIGNAL
    # bank_deposit_ratio_log range: log1p(0)=0 to log1p(543988)=13.2
    # Normalize to 0-14 range (config FIXED_NORM_BOUNDS updated)
    if "bank_deposit_ratio_log" in df.columns:
        bdr_log = df["bank_deposit_ratio_log"].fillna(0)
    elif "bank_deposit_ratio" in df.columns:
        bdr_log = pd.Series(
            np.log1p(np.maximum(df["bank_deposit_ratio"].fillna(0), 0)),
            index=df.index,
        )
    else:
        bdr_log = pd.Series(0.0, index=df.index)

    # FIXED: was norm01(bdr_log, 0.0, 6.2) — max was wrong (6.2 = log1p(500))
    # Actual max = log1p(543988) = 13.2, use 14.0 for headroom
    df["bank_deposit_block_score"] = norm01(bdr_log, 0.0, 14.0)

    # BLOCK 3 — Z-SCORE COMPOSITE
    if "composite_z_score" in df.columns:
        df["zscore_block_score"] = norm01(
            df["composite_z_score"].fillna(0), -3.0, 3.0
        )
    else:
        df["zscore_block_score"] = np.float32(0.5)

    # BLOCK 4 — NETWORK SIGNAL
    df["network_block_score"] = (
        df.get("network_risk_score", pd.Series(0.21, index=df.index))
        .fillna(0.21)
        .astype("float32")
    )

    # BLOCK 5 — WITHHOLDING SIGNAL (INVERTED)
    if "federal_withheld" in df.columns:
        fw = df["federal_withheld"].fillna(0).clip(0, 50_000)
        df["withholding_block_score"] = (
            1.0 - norm01(fw, 0.0, 20_000.0)
        ).astype("float32")
    elif "withholding_rate" in df.columns:
        wr = df["withholding_rate"].fillna(0)
        df["withholding_block_score"] = (
            1.0 - norm01(wr, 0.0, 0.30)
        ).astype("float32")
    else:
        df["withholding_block_score"] = np.float32(0.5)

    # BLOCK 6 — FORENSIC SIGNAL
    df["forensic_block_score"] = (
        df.get("forensic_risk_score", pd.Series(0.0, index=df.index))
        .fillna(0)
        .astype("float32")
    )

    # Fix 6: dead constant blocks REMOVED entirely
    # lifestyle_block_score, tax_burden_block_score, deduction_block_score
    # are gone — they were constant 0.0 and added noise to feature space

    df["flag_block_score"] = stable_norm01(
        df["total_flag_count"].fillna(0), "total_flag_count"
    )

    # MASTER FRAUD PROPENSITY
    block_weights = {
        "irs_block_score_nl":       0.50,
        "bank_deposit_block_score": 0.20,
        "zscore_block_score":       0.12,
        "network_block_score":      0.10,
        "withholding_block_score":  0.05,
        "forensic_block_score":     0.03,
    }

    master  = pd.Series(0.0, index=df.index, dtype=np.float64)
    total_w = 0.0
    for block, weight in block_weights.items():
        if block in df.columns:
            master  += df[block].fillna(0).astype(np.float64) * weight
            total_w += weight

    master_base = (master / max(total_w, EPS)).clip(0.0, 1.0)

    # Interaction terms
    flag_irs_high = df.get(
        "flag_irs_high",
        pd.Series(
            (df.get("irs_risk_score", pd.Series(0, index=df.index))
             .fillna(0) > 67).astype(float),
            index=df.index,
        ),
    ).fillna(0)

    flag_bdr_extreme = df.get(
        "flag_bank_deposit_extreme",
        pd.Series(
            (df.get("bank_deposit_ratio", pd.Series(0, index=df.index))
             .fillna(0) > 43).astype(float),
            index=df.index,
        ),
    ).fillna(0)

    # Fix 5: safe fallback chain for flag_low_wh
    if "flag_low_withholding" in df.columns:
        flag_low_wh = df["flag_low_withholding"].fillna(0)
    elif "withholding_rate" in df.columns:
        flag_low_wh = (
            df["withholding_rate"].fillna(0) < 0.03
        ).astype(float)
    else:
        flag_low_wh = pd.Series(0.0, index=df.index)

    interaction_irs_bdr = flag_irs_high * flag_bdr_extreme * 0.12
    interaction_irs_wh  = flag_irs_high * flag_low_wh      * 0.08

    flag_boost = pd.Series(0.0, index=df.index)
    for flag_col, boost_val in [
        ("flag_irs_high",             0.04),
        ("flag_bank_deposit_extreme", 0.04),
        ("flag_low_withholding",      0.02),
        ("flag_phantom_business",     0.01),
        ("flag_hobby_loss_risk",      0.01),
    ]:
        if flag_col in df.columns:
            flag_boost += df[flag_col].fillna(0).astype(float) * boost_val

    df["master_fraud_propensity"] = (
        master_base
        + interaction_irs_bdr
        + interaction_irs_wh
        + flag_boost
    ).clip(0.0, 1.0).astype("float32")

    # FRAUD-TYPE AWARE SIGNALS
    if "deduction_taken" in df.columns and "agi" in df.columns:
        df["deduction_fraud_signal"] = (
            df["deduction_taken"].fillna(0) / (df["agi"].abs() + EPS)
        ).clip(0, 5).astype("float32")

    bdr_log_col = df.get(
        "bank_deposit_ratio_log",
        pd.Series(
            np.log1p(np.maximum(
                df.get("bank_deposit_ratio", pd.Series(0, index=df.index))
                .fillna(0), 0
            )),
            index=df.index,
        ),
    )
    df["income_hiding_signal"] = (
        bdr_log_col * (1.0 - norm01(df["agi"].fillna(0), 0, 200_000))
    ).clip(0, 10).astype("float32")

    if "gross_receipts" in df.columns and "sch_c_total_expenses" in df.columns:
        df["business_suppression_signal"] = (
            df["sch_c_total_expenses"].fillna(0) /
            (df["gross_receipts"].fillna(0) + EPS)
        ).clip(0, 5).astype("float32")

    # FIXED VOCABULARY ONE-HOT ENCODING
    for state in VALID_STATES:
        safe = state.lower().replace(" ", "_")
        df[f"state_{safe}"] = (df["state"] == state).astype("int8")

    for tt in VALID_TAXPAYER_TYPES:
        df[f"tt_{tt}"] = (df["taxpayer_type"] == tt).astype("int8")

    if "age_group" in df.columns:
        for ag in VALID_AGE_GROUPS:
            df[f"age_{ag}"] = (df["age_group"] == ag).astype("int8")

    return df


def separation_report(df: pd.DataFrame) -> float:
    score = df["master_fraud_propensity"]
    fraud = score[df["fraud_label"] == 1]
    clean = score[df["fraud_label"] == 0]

    fraud_mean = fraud.mean()
    clean_mean = clean.mean()
    sep_ratio  = fraud_mean / max(clean_mean, EPS)

    clean_p50       = clean.quantile(0.50)
    clean_p75       = clean.quantile(0.75)
    fraud_beats_p50 = (fraud > clean_p50).mean()
    fraud_beats_p75 = (fraud > clean_p75).mean()

    thresholds = np.linspace(score.min(), score.max(), 500)
    ks_stat    = float(np.max(np.abs(
        np.mean(fraud.values[:, None] <= thresholds, axis=0) -
        np.mean(clean.values[:, None] <= thresholds, axis=0)
    )))

    print(f"    Fraud mean       : {fraud_mean:.4f}")
    print(f"    Clean mean       : {clean_mean:.4f}")
    print(f"    Separation ratio : {sep_ratio:.2f}x  "
          f"{'[OK]' if sep_ratio > 1.3 else '[WARN]'}")
    print(f"    Fraud > clean p50: {fraud_beats_p50*100:.1f}%  "
          f"{'[OK]' if fraud_beats_p50 > 0.60 else '[WARN]'}")
    print(f"    Fraud > clean p75: {fraud_beats_p75*100:.1f}%  "
          f"{'[OK]' if fraud_beats_p75 > 0.35 else '[WARN]'}")
    print(f"    KS statistic     : {ks_stat:.4f}  "
          f"{'[OK]' if ks_stat > 0.10 else '[WARN]'}")

    for block in [
        "irs_block_score_nl", "bank_deposit_block_score",
        "zscore_block_score", "network_block_score", "withholding_block_score",
    ]:
        if block in df.columns:
            bf = df.loc[df["fraud_label"] == 1, block].mean()
            bc = df.loc[df["fraud_label"] == 0, block].mean()
            print(f"    {block:<34}: fraud={bf:.4f}  clean={bc:.4f}  "
                  f"ratio={bf/max(bc,EPS):.2f}x")

    return sep_ratio


def run():
    print("=" * 65)
    print("PHASE 8 -- COMPOSITE SCORES")
    print("=" * 65)

    for year in YEARS:
        p = ENGINEERED_DIR / f"year_{year}_engineered.parquet"
        if not p.exists():
            print(f"  year_{year}: MISSING -- skipped")
            continue

        print(f"\n  year_{year}: loading...", end="")
        df = pd.read_parquet(p)

        # Fix 6: drop dead constant blocks if they exist from old runs
        drop_cols = [
            c for c in df.columns
            if c.endswith(("_block_score", "_block_score_nl"))
            or c in (
                "master_fraud_propensity",
                "deduction_fraud_signal",
                "income_hiding_signal",
                "business_suppression_signal",
                "lifestyle_block_score",     # Fix 6: explicitly listed
                "tax_burden_block_score",    # Fix 6: explicitly listed
                "deduction_block_score",     # Fix 6: explicitly listed
            )
        ]
        df = df.drop(columns=drop_cols, errors="ignore")

        n_before = df.shape[1]
        print(f" {len(df):,} rows")

        df = add_composite_scores(df)

        print(f"    New features     : {df.shape[1] - n_before}")
        print(f"    Total cols       : {df.shape[1]}")
        separation_report(df)

        df.to_parquet(p, index=False)
        print(f"    Saved            : {p.name}")
        del df; gc.collect()

    print(f"\n{'=' * 65}")
    print("PHASE 8 COMPLETE")


if __name__ == "__main__":
    run()