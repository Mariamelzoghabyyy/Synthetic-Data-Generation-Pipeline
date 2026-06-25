# 04_forensic_features.py
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — FORENSIC ACCOUNTING FEATURES
#
# Key changes in this version:
#   - cash_t_method_gap and unexplained_wealth_index now use the uncapped
#     bank_deposit_ratio (0-500) so values are no longer 25x too small
#   - FIXED_NORM_BOUNDS for cash_t_method_gap_ratio updated to (-1, 300)
#   - Benford first-digit extraction stabilized with log rounding
#   - stable_norm01 bypasses normalization for flag_/has_ columns
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
    ENGINEERED_DIR, YEARS, SCH_C_COLS,
    BENFORD_EXPECTED, FIXED_NORM_BOUNDS, EPS,
)


def vectorized_benford_deviation(matrix: np.ndarray) -> np.ndarray:
    """
    Fully vectorized Benford's Law chi-squared deviation.
    Mathematical first-digit extraction — no string conversion.

    Stabilized: rounds log10 integer part to nearest integer when within
    1e-10, preventing float precision errors on exact powers of 10
    (e.g. log10(1000) = 2.9999... -> floor gives 2 not 3 on some platforms).

    Rows with fewer than 3 valid values return 0 (insufficient data).
    """
    abs_m = np.abs(matrix)
    valid = (abs_m >= 1.0) & ~np.isnan(abs_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_v        = np.log10(np.where(valid, abs_m, 1.0))
        log_int      = np.floor(log_v + 1e-10)           # stabilized floor
        frac_part    = log_v - log_int
        first_digits = np.floor(10.0 ** frac_part + 1e-10).astype(np.int32)

    first_digits = np.where(valid, np.clip(first_digits, 1, 9), 0)

    row_counts = np.zeros((matrix.shape[0], 9), dtype=np.float32)
    for d in range(1, 10):
        row_counts[:, d - 1] = (first_digits == d).sum(axis=1)

    row_totals = row_counts.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        observed = row_counts / (row_totals + EPS)
        chi_sq   = ((observed - BENFORD_EXPECTED) ** 2 /
                    BENFORD_EXPECTED).sum(axis=1)

    return np.where(row_totals.squeeze() >= 3, chi_sq, 0.0).astype(np.float32)


def vectorized_round_number_ratio(matrix: np.ndarray) -> np.ndarray:
    """
    Vectorized round number detection without string conversion.
    Flags values divisible by 100 or 1000.
    """
    abs_m    = np.abs(np.nan_to_num(matrix, nan=0.0)).astype(np.int64)
    valid    = abs_m != 0
    is_round = ((abs_m % 1000 == 0) | (abs_m % 100 == 0)) & valid

    row_rounds = is_round.sum(axis=1)
    row_totals = valid.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = row_rounds / (row_totals + EPS)

    return np.where(row_totals > 0, ratio, 0.0).astype(np.float32)


def stable_norm01(series: pd.Series, name: str) -> pd.Series:
    """
    Normalize to [0,1] using fixed training-set bounds.
    Binary/flag features bypass normalization — already in {0,1} and
    rare flags have p99=0 which would zero out the entire column.
    """
    if name.startswith(("flag_", "has_")) or series.nunique() <= 2:
        return series.fillna(0.0).astype("float32")

    if name in FIXED_NORM_BOUNDS:
        lo, hi = FIXED_NORM_BOUNDS[name]
    else:
        lo = float(series.quantile(0.01))
        hi = float(series.quantile(0.99))

    if hi <= lo:
        return pd.Series(0.0, index=series.index, dtype=np.float32)

    return (
        (series - lo) / (hi - lo + EPS)
    ).clip(0.0, 1.0).astype("float32")


def add_forensic_features(df: pd.DataFrame) -> pd.DataFrame:
    sch_c_present = [c for c in SCH_C_COLS if c in df.columns]

    # ── Benford's Law ─────────────────────────────────────────────────────────
    if len(sch_c_present) >= 3:
        matrix = df[sch_c_present].values
        df["benford_deviation_expenses"] = vectorized_benford_deviation(matrix)
    else:
        df["benford_deviation_expenses"] = np.float32(0.0)

    # ── Round number ratio ────────────────────────────────────────────────────
    if len(sch_c_present) >= 2:
        matrix = df[sch_c_present].values
        df["round_number_ratio"] = vectorized_round_number_ratio(matrix)
    else:
        df["round_number_ratio"] = np.float32(0.0)

    # ── Net worth growth proxy ────────────────────────────────────────────────
    df["net_worth_growth_proxy"] = (
        df["agi"] -
        df["total_tax_liability"].fillna(0) -
        df["absolute_lifestyle_spend"].fillna(0)
    ).clip(-2_000_000, 2_000_000).astype("float32")

    # ── IRS Cash T Method proxy ───────────────────────────────────────────────
    # bank_deposit_ratio now 0-500 so estimated_deposits and gap are
    # correctly scaled (previously 25x too small due to clip at 20).
    estimated_deposits = df["bank_deposit_ratio"].fillna(1) * df["agi"]
    df["cash_t_method_gap"] = (
        estimated_deposits - df["agi"]
    ).clip(-1_000_000, 25_000_000).astype("float32")

    df["cash_t_method_gap_ratio"] = (
        df["cash_t_method_gap"] / (df["agi"].abs() + EPS)
    ).clip(-5, 300).astype("float32")

    # ── Lifestyle asset score ─────────────────────────────────────────────────
    df["lifestyle_asset_score"] = (
        0.40 * df["lifestyle_income_ratio"].fillna(0).clip(0, 5) +
        0.35 * df["bank_deposit_ratio"].fillna(0).clip(0, 5) +
        0.25 * df["utility_income_ratio"].fillna(0).clip(0, 2) * 2
    ).clip(0, 5).astype("float32")

    # ── Unexplained wealth ────────────────────────────────────────────────────
    total_observable = (
        df["absolute_lifestyle_spend"].fillna(0) +
        (df["bank_deposit_ratio"].fillna(0) * df["agi"])
    )
    df["unexplained_wealth_index"] = (
        np.maximum(0, total_observable - df["agi"])
    ).clip(0, 25_000_000).astype("float32")

    df["unexplained_wealth_ratio"] = (
        df["unexplained_wealth_index"] / (df["agi"].abs() + EPS)
    ).clip(0, 300).astype("float32")

    # ── DIF Proxy Score ───────────────────────────────────────────────────────
    dif_components, dif_weights = [], []
    for col, w in [
        ("z_deduction_income_ratio",   0.25),
        ("z_lifestyle_income_ratio",   0.20),
        ("tax_rate_gap",               0.20),
        ("z_sch_c_to_revenue_ratio",   0.15),
        ("benford_deviation_expenses", 0.10),
        ("round_number_ratio",         0.10),
    ]:
        if col in df.columns:
            dif_components.append(stable_norm01(df[col].fillna(0), col) * w)
            dif_weights.append(w)

    df["dif_proxy_score"] = (
        sum(dif_components) / max(sum(dif_weights), EPS)
    ).clip(0, 1).astype("float32") if dif_components else np.float32(0.0)

    # ── Forensic Risk Score ───────────────────────────────────────────────────
    forensic_components, forensic_weights = [], []
    for col, w in [
        ("benford_deviation_expenses", 0.25),
        ("round_number_ratio",         0.20),
        ("unexplained_wealth_ratio",   0.25),
        ("cash_t_method_gap_ratio",    0.20),
        ("lifestyle_asset_score",      0.10),
    ]:
        if col in df.columns:
            forensic_components.append(stable_norm01(df[col].fillna(0), col) * w)
            forensic_weights.append(w)

    df["forensic_risk_score"] = (
        sum(forensic_components) / max(sum(forensic_weights), EPS)
    ).clip(0, 1).astype("float32") if forensic_components else np.float32(0.0)

    return df


def run():
    print("=" * 65)
    print("PHASE 5 -- FORENSIC ACCOUNTING FEATURES")
    print("=" * 65)

    for year in YEARS:
        p = ENGINEERED_DIR / f"year_{year}_engineered.parquet"
        if not p.exists():
            print(f"  year_{year}: MISSING -- skipped")
            continue

        print(f"\n  year_{year}: loading...", end="")
        df       = pd.read_parquet(p)
        n_before = df.shape[1]
        print(f" {len(df):,} rows")

        df = add_forensic_features(df)

        print(f"    New features : {df.shape[1] - n_before}")
        print(f"    Total cols   : {df.shape[1]}")
        df.to_parquet(p, index=False)
        print(f"    Saved        : {p.name}")
        del df; gc.collect()

    print(f"\n{'=' * 65}")
    print("PHASE 5 COMPLETE")


if __name__ == "__main__":
    run()