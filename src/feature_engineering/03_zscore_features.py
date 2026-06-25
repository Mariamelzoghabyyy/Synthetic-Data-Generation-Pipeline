# 03_zscore_features.py
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Z-SCORE AND TARGET ENCODING FEATURES
#
# Key changes in this version:
#   - composite_z_score uses updated Z_SCORE_WEIGHTS from config
#     (irs_risk_score weight raised 0.20->0.40, bank 0.20->0.25)
#   - bank_deposit_ratio_log and zone_risk added to feats_to_rank
#     so pct_ versions are computed for these new columns
#   - Merge-added columns tracked explicitly before/after merge
#     to prevent suffix-based drop from removing legitimate features
#   - Single unified groupby rank call for all features (10x speedup)
# ═══════════════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pickle
import gc
import warnings
warnings.filterwarnings("ignore")

from config import (
    ENGINEERED_DIR, PEER_DIR, YEARS,
    Z_SCORE_FEATURES, Z_SCORE_WEIGHTS, EPS,
)

# Additional features to rank within peer groups beyond Z_SCORE_FEATURES
EXTRA_RANK_FEATURES = [
    "bank_deposit_ratio_log",
    "zone_risk",
    "irs_risk_score_sq",
    "withholding_rate",
]


def run():
    print("=" * 65)
    print("PHASE 4 -- Z-SCORE AND TARGET ENCODING FEATURES")
    print("=" * 65)

    peer_stats = pd.read_parquet(PEER_DIR / "peer_stats.parquet")
    print(f"  Peer groups loaded: {len(peer_stats):,}")

    with open(PEER_DIR / "group_fraud_rates.pkl", "rb") as f:
        group_fraud = pickle.load(f)

    global_mean     = group_fraud["global_mean"]
    state_fraud_map = dict(zip(
        group_fraud["state"]["state"],
        group_fraud["state"]["state_fraud_rate"],
    ))
    ttype_fraud_map = dict(zip(
        group_fraud["ttype"]["taxpayer_type"],
        group_fraud["ttype"]["ttype_fraud_rate"],
    ))
    band_fraud_map  = dict(zip(
        group_fraud["income_band"]["income_band"],
        group_fraud["income_band"]["band_fraud_rate"],
    ))
    year_fraud_map  = dict(zip(
        group_fraud["year"]["tax_year"],
        group_fraud["year"]["year_fraud_rate"],
    ))

    print(f"  Global fraud mean   : {global_mean:.4f}")

    for year in YEARS:
        p = ENGINEERED_DIR / f"year_{year}_engineered.parquet"
        if not p.exists():
            print(f"  year_{year}: MISSING -- skipped")
            continue

        print(f"\n  year_{year}: loading...", end="")
        df       = pd.read_parquet(p)
        n_before = df.shape[1]
        print(f" {len(df):,} rows")

        # ── Merge peer stats ──────────────────────────────────────────────────
        merge_cols        = ["income_band", "taxpayer_type", "state"]
        cols_before_merge = set(df.columns)
        df                = df.merge(peer_stats, on=merge_cols, how="left")
        peer_added_cols   = set(df.columns) - cols_before_merge

        # ── Z-scores ──────────────────────────────────────────────────────────
        z_cols        = []
        feats_to_rank = []

        for feat in Z_SCORE_FEATURES:
            if feat not in df.columns:
                continue
            mean_col = f"{feat}_mean"
            std_col  = f"{feat}_std"
            if mean_col not in df.columns or std_col not in df.columns:
                continue

            z_col = f"z_{feat}"
            df[z_col] = (
                (df[feat] - df[mean_col]) /
                df[std_col].replace(0, np.nan).fillna(1.0)
            ).clip(-5, 5).astype("float32")
            z_cols.append(z_col)
            feats_to_rank.append(feat)

        # Add extra features to rank list (if present in df)
        for feat in EXTRA_RANK_FEATURES:
            if feat in df.columns and feat not in feats_to_rank:
                feats_to_rank.append(feat)

        # ── TRUE rank-based percentiles — SINGLE unified groupby call ─────────
        # Replaces N individual groupby rank calls with one parallelized call.
        if feats_to_rank:
            df_pcts = (
                df.groupby(merge_cols)[feats_to_rank]
                .rank(pct=True, na_option="keep") * 100
            ).fillna(50.0).astype("float32")

            for feat in feats_to_rank:
                df[f"pct_{feat}"] = df_pcts[feat]

        # ── Z-score summary stats ─────────────────────────────────────────────
        if z_cols:
            df["n_z_above_2"] = (df[z_cols].abs() > 2.0).sum(axis=1).astype("int8")
            df["n_z_above_3"] = (df[z_cols].abs() > 3.0).sum(axis=1).astype("int8")
            df["max_z_score"] = df[z_cols].abs().max(axis=1).astype("float32")
            df["mean_abs_z"]  = df[z_cols].abs().mean(axis=1).astype("float32")

            # Weighted composite z-score using updated Z_SCORE_WEIGHTS
            # irs_risk_score weight = 0.40, bank_deposit_ratio = 0.25
            comp  = np.zeros(len(df), dtype=np.float32)
            tot_w = 0.0
            for feat, weight in Z_SCORE_WEIGHTS.items():
                zc = f"z_{feat}"
                if zc in df.columns:
                    comp  += df[zc].fillna(0).values * weight
                    tot_w += weight
            df["composite_z_score"] = (
                comp / max(tot_w, EPS)
            ).clip(-5, 5).astype("float32")

        # ── Target encoding ───────────────────────────────────────────────────
        df["state_fraud_rate"] = (
            df["state"].map(state_fraud_map).fillna(global_mean)
        ).astype("float32")
        df["ttype_fraud_rate"] = (
            df["taxpayer_type"].map(ttype_fraud_map).fillna(global_mean)
        ).astype("float32")
        df["band_fraud_rate"]  = (
            df["income_band"].map(band_fraud_map).fillna(global_mean)
        ).astype("float32")
        df["year_fraud_rate"]  = (
            df["tax_year"].map(year_fraud_map).fillna(global_mean)
        ).astype("float32")
        df["peer_fraud_rate"]  = (
            df["peer_fraud_rate"].fillna(global_mean)
        ).astype("float32")

        # ── Drop intermediate peer stat columns ───────────────────────────────
        # Track only columns added by the merge — prevents removing
        # legitimate features that happen to share suffix patterns.
        drop_cols = [
            c for c in peer_added_cols
            if (
                c.endswith(("_mean", "_std", "_median", "_p95"))
                and not c.startswith(("z_", "pct_"))
                and c not in (
                    "composite_z_score",
                    "peer_fraud_rate",
                    "peer_group_size",
                    "raw_peer_fraud_rate",
                )
            )
        ]
        df = df.drop(columns=drop_cols, errors="ignore")

        n_new = df.shape[1] - n_before
        print(f"    New features : {n_new}  |  Total cols : {df.shape[1]}")
        df.to_parquet(p, index=False)
        print(f"    Saved        : {p.name}")
        del df; gc.collect()

    print(f"\n{'=' * 65}")
    print("PHASE 4 COMPLETE")


if __name__ == "__main__":
    run()