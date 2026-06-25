# 05_network_features.py
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — NETWORK / RELATIONAL FEATURES
#
# No functional changes from previous reviewed version.
# Retains: global emp_std computed once, graceful fallback, zero-std guard.
# contagion_score uses updated composite_z_score (new weights from Phase 4).
# ═══════════════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

from config import ENGINEERED_DIR, PEER_DIR, YEARS, EPS


def run():
    print("=" * 65)
    print("PHASE 6 -- NETWORK / RELATIONAL FEATURES")
    print("=" * 65)

    emp_path = PEER_DIR / "employer_stats.parquet"
    if emp_path.exists():
        employer_stats   = pd.read_parquet(emp_path)
        global_emp_fraud = float(employer_stats["employer_fraud_rate"].mean())
        emp_std          = float(employer_stats["employer_fraud_rate"].std())
        emp_std          = emp_std if emp_std > EPS else 1.0
        print(f"  Employer stats   : {len(employer_stats):,} employers")
        print(f"  Mean fraud rate  : {global_emp_fraud:.4f}")
        print(f"  Std  fraud rate  : {emp_std:.4f}")
    else:
        employer_stats   = None
        global_emp_fraud = 0.21
        emp_std          = 1.0
        print("  WARNING: employer_stats.parquet not found")
        print("           Using global defaults (fraud=0.21, std=1.0)")

    for year in YEARS:
        p = ENGINEERED_DIR / f"year_{year}_engineered.parquet"
        if not p.exists():
            print(f"\n  year_{year}: MISSING -- skipped")
            continue

        print(f"\n  year_{year}: loading...", end="")
        df       = pd.read_parquet(p)
        n_before = df.shape[1]
        print(f" {len(df):,} rows")

        if employer_stats is not None and "employer_id" in df.columns:
            df = df.merge(
                employer_stats[[
                    "employer_id",
                    "employer_fraud_rate",
                    "employer_n_employees",
                ]],
                on="employer_id",
                how="left",
            )
            df["employer_fraud_rate"]  = (
                df["employer_fraud_rate"].fillna(global_emp_fraud).astype("float32")
            )
            df["employer_n_employees"] = (
                df["employer_n_employees"].fillna(0).astype("int32")
            )
            df["employer_fraud_z_score"] = (
                (df["employer_fraud_rate"] - global_emp_fraud) / emp_std
            ).clip(-5, 5).astype("float32")
            df["high_risk_employer"] = (
                df["employer_fraud_rate"] > 0.30
            ).astype("int8")
        else:
            df["employer_fraud_rate"]    = np.float32(global_emp_fraud)
            df["employer_n_employees"]   = np.int32(0)
            df["employer_fraud_z_score"] = np.float32(0.0)
            df["high_risk_employer"]     = np.int8(0)

        df["network_risk_score"] = (
            0.50 * df["employer_fraud_rate"].fillna(global_emp_fraud) +
            0.30 * df["peer_fraud_rate"].fillna(0.21) +
            0.20 * df["state_fraud_rate"].fillna(0.21)
        ).clip(0, 1).astype("float32")

        # contagion_score uses updated composite_z_score (Phase 4 new weights)
        df["contagion_score"] = (
            0.40 * df["employer_fraud_z_score"].fillna(0).clip(0, 5) +
            0.35 * df.get(
                "z_lifestyle_income_ratio",
                pd.Series(0, index=df.index),
            ).fillna(0).clip(0, 5) +
            0.25 * df.get(
                "composite_z_score",
                pd.Series(0, index=df.index),
            ).fillna(0).clip(0, 5)
        ).clip(0, 5).astype("float32")

        print(f"    New features : {df.shape[1] - n_before}")
        print(f"    Total cols   : {df.shape[1]}")
        df.to_parquet(p, index=False)
        print(f"    Saved        : {p.name}")
        del df; gc.collect()

    print(f"\n{'=' * 65}")
    print("PHASE 6 COMPLETE")


if __name__ == "__main__":
    run()