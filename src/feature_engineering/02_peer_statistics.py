# 02_peer_statistics.py
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — PEER GROUP STATISTICS
#
# Key changes in this version:
#   - PEER_FEATURES now includes bank_deposit_ratio_log and zone_risk
#     (these are new columns from Phase 1 raw signal features)
#   - pq.read_schema() used for column introspection (no empty df read)
#   - year_fraud_rate computed from train_df directly (no second file reads)
#   - Global stds precomputed once before std_cols loop
#   - Single Cythonized groupby quantile call for p95 (no per-lambda)
#   - M-estimate smoothing on all fraud rates
# ═══════════════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pickle
import gc
import warnings
warnings.filterwarnings("ignore")

from config import (
    ENGINEERED_DIR, PEER_DIR, TRAIN_YEARS, YEARS,
    Z_SCORE_FEATURES, M_WEIGHT,
)

# PEER_FEATURES: columns used for peer group statistics.
# Includes new raw signal features from Phase 1.
PEER_FEATURES = Z_SCORE_FEATURES + [
    "sch_c_total_expenses",
    "income_stream_count",
    "absolute_lifestyle_gap",
    "bank_deposit_gap_ratio",
    "bank_deposit_ratio_log",   # NEW: log-normalized bank deposit
    "zone_risk",                # NEW: inverted zone encoding
]


def smoothed_fraud_rate(
    series: pd.Series,
    global_mean: float,
    m: float = M_WEIGHT,
) -> float:
    """
    M-estimate smoothed fraud rate for a group.
    Pulls small groups toward global_mean.
    Large groups (n >> m) are barely affected.
    Small groups (n << m) pulled strongly toward global mean.
    """
    n   = len(series)
    raw = series.mean()
    return (n * raw + m * global_mean) / (n + m)


def run():
    print("=" * 65)
    print("PHASE 3 -- PEER GROUP STATISTICS")
    print("=" * 65)
    print(f"  Training years only : {TRAIN_YEARS}")
    print(f"  Leakage prevention  : m-estimate smoothing (m={M_WEIGHT})")

    # ── Load training data ────────────────────────────────────────────────────
    frames = []
    for year in TRAIN_YEARS:
        p = ENGINEERED_DIR / f"year_{year}_engineered.parquet"
        if not p.exists():
            print(f"  year_{year}: MISSING -- skipped")
            continue

        available_cols = set(pq.read_schema(p).names)
        cols_needed    = list(
            {"income_band", "taxpayer_type", "state",
             "employer_id", "fraud_label", "tax_year"} |
            {f for f in PEER_FEATURES if f in available_cols}
        )

        df = pd.read_parquet(p, columns=cols_needed)
        frames.append(df)
        print(f"  Loaded year_{year}: {len(df):,} rows")
        del df; gc.collect()

    if not frames:
        print("  [FAIL] No training data found -- Phases 1 and 2 must run first")
        return

    train_df = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()
    print(f"\n  Combined training rows : {len(train_df):,}")

    global_fraud_mean = float(train_df["fraud_label"].mean())
    print(f"  Global fraud rate     : {global_fraud_mean:.4f}")

    # ── Peer group statistics ─────────────────────────────────────────────────
    print("\n  Computing peer group statistics...")
    group_cols   = ["income_band", "taxpayer_type", "state"]
    simple_feats = [f for f in PEER_FEATURES if f in train_df.columns]

    agg_kwargs: dict = {
        "peer_group_size":     ("fraud_label", "count"),
        "raw_peer_fraud_rate": ("fraud_label", "mean"),
    }
    for feat in simple_feats:
        agg_kwargs[f"{feat}_mean"]   = (feat, "mean")
        agg_kwargs[f"{feat}_std"]    = (feat, "std")
        agg_kwargs[f"{feat}_median"] = (feat, "median")

    peer_stats = train_df.groupby(group_cols).agg(**agg_kwargs).reset_index()

    # 95th percentile — single Cythonized call (20x faster than per-lambda)
    if simple_feats:
        print("  Computing peer group 95th percentiles (vectorized)...")
        p95_stats  = (
            train_df
            .groupby(group_cols)[simple_feats]
            .quantile(0.95)
            .reset_index()
        )
        rename_dict = {feat: f"{feat}_p95" for feat in simple_feats}
        p95_stats   = p95_stats.rename(columns=rename_dict)
        peer_stats  = peer_stats.merge(p95_stats, on=group_cols, how="left")

    # M-estimate smoothing on fraud rate
    peer_stats["peer_fraud_rate"] = (
        (peer_stats["peer_group_size"] * peer_stats["raw_peer_fraud_rate"]) +
        (M_WEIGHT * global_fraud_mean)
    ) / (peer_stats["peer_group_size"] + M_WEIGHT)

    # Fill std nulls for single-member groups using pre-computed global stds
    global_stds = (
        train_df[simple_feats].std().to_dict() if simple_feats else {}
    )
    std_cols = [c for c in peer_stats.columns if c.endswith("_std")]
    for col in std_cols:
        base_feat  = col.replace("_std", "")
        global_std = float(global_stds.get(base_feat, 1.0))
        peer_stats[col] = peer_stats[col].fillna(global_std)

    peer_stats.to_parquet(PEER_DIR / "peer_stats.parquet", index=False)
    print(f"  Peer groups saved: {len(peer_stats):,}")

    # ── Employer fraud rates ──────────────────────────────────────────────────
    print("\n  Computing employer fraud rates...")
    if "employer_id" in train_df.columns:
        emp_agg = train_df.groupby("employer_id").agg(
            raw_employer_fraud_rate=("fraud_label", "mean"),
            employer_n_employees   =("fraud_label", "count"),
        ).reset_index()

        emp_agg["employer_fraud_rate"] = (
            (emp_agg["employer_n_employees"] * emp_agg["raw_employer_fraud_rate"]) +
            (M_WEIGHT * global_fraud_mean)
        ) / (emp_agg["employer_n_employees"] + M_WEIGHT)

        emp_agg.to_parquet(PEER_DIR / "employer_stats.parquet", index=False)
        print(f"  Employers saved: {len(emp_agg):,}")

    # ── Group-level fraud rates (from already-loaded train_df) ────────────────
    print("\n  Computing group fraud rates...")

    state_fraud = (
        train_df.groupby("state")["fraud_label"]
        .apply(lambda x: smoothed_fraud_rate(x, global_fraud_mean))
        .reset_index()
        .rename(columns={"fraud_label": "state_fraud_rate"})
    )
    ttype_fraud = (
        train_df.groupby("taxpayer_type")["fraud_label"]
        .apply(lambda x: smoothed_fraud_rate(x, global_fraud_mean))
        .reset_index()
        .rename(columns={"fraud_label": "ttype_fraud_rate"})
    )
    band_fraud = (
        train_df.groupby("income_band")["fraud_label"]
        .apply(lambda x: smoothed_fraud_rate(x, global_fraud_mean))
        .reset_index()
        .rename(columns={"fraud_label": "band_fraud_rate"})
    )
    year_fraud = (
        train_df.groupby("tax_year")["fraud_label"]
        .apply(lambda x: smoothed_fraud_rate(x, global_fraud_mean))
        .reset_index()
        .rename(columns={"fraud_label": "year_fraud_rate"})
    )

    group_fraud = {
        "state":       state_fraud,
        "ttype":       ttype_fraud,
        "income_band": band_fraud,
        "year":        year_fraud,
        "global_mean": global_fraud_mean,
    }
    with open(PEER_DIR / "group_fraud_rates.pkl", "wb") as f:
        pickle.dump(group_fraud, f)

    print("  Saved: group_fraud_rates.pkl")
    print(f"\n{'=' * 65}")
    print("PHASE 3 COMPLETE")
    for f in sorted(PEER_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")

    del train_df; gc.collect()


if __name__ == "__main__":
    run()