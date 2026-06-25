# 08_feature_selection.py
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 9 — FINAL OUTPUT GENERATION
#
# This version does NO selection/dropping. It takes the full engineered
# files (which already contain ALL raw columns merged in Phase 1 + ALL
# engineered features from Phases 2-8) and writes them to the ML_READY
# output directories, preserving the structure:
#
#   ml_ready/by_year/year_YYYY.parquet         (all data for that year)
#   ml_ready/by_state/STATE.parquet             (all years for that state)
#   ml_ready/by_state_by_year/STATE_YYYY.parquet (state + year combo)
#
# Each output file contains:
#   • ALL original raw columns (preserved from Phase 1 merge)
#   • ALL engineered features (no selection, no dropping)
#   • fraud_label, tax_year, state, taxpayer_type metadata
# ═══════════════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import gc
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from config import (
    ENGINEERED_DIR,
    ML_READY_DIR,
    YEARS,
    VALID_STATES,
    RANDOM_STATE,
)


def to_safe(state: str) -> str:
    return state.strip().lower().replace(" ", "_")


def run():

    print("=" * 65)
    print("PHASE 9 — FINAL OUTPUT GENERATION")
    print("=" * 65)
    print(
        "\n  This phase reads the FULL engineered files (which contain\n"
        "  ALL raw columns + ALL engineered features) and writes them\n"
        "  to the ML_READY directory structure.\n"
        "\n  NO feature selection is applied — all columns are preserved.\n"
    )

    # ── Create output directories ────────────────────────────────────────────
    ML_BY_YEAR         = ML_READY_DIR / "by_year"
    ML_BY_STATE        = ML_READY_DIR / "by_state"
    ML_BY_STATE_YEAR   = ML_READY_DIR / "by_state_by_year"

    for d in [ML_BY_YEAR, ML_BY_STATE, ML_BY_STATE_YEAR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Write by_year files ──────────────────────────────────────────
    print("\n  Writing by_year files...\n")

    for year in YEARS:
        p_in  = ENGINEERED_DIR / f"year_{year}_engineered.parquet"
        p_out = ML_BY_YEAR     / f"year_{year}.parquet"

        if not p_in.exists():
            print(f"  year_{year}: engineered file MISSING — run Phases 1-8 first")
            continue

        df = pd.read_parquet(p_in)

        # Ensure critical metadata columns exist
        for col in ["fraud_label", "tax_year", "state", "taxpayer_type"]:
            if col not in df.columns:
                df[col] = np.int8(0) if col == "fraud_label" else "UNKNOWN"

        df.to_parquet(p_out, index=False)
        size_mb = p_out.stat().st_size / 1e6
        print(f"  year_{year}.parquet : {len(df):,} rows × {df.shape[1]} cols ({size_mb:.0f} MB)")
        del df; gc.collect()

    # ── Step 2: Load all data for state splits ───────────────────────────────
    print("\n  Loading all years for state splitting...\n")

    all_dfs = []
    for year in YEARS:
        p = ML_BY_YEAR / f"year_{year}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            all_dfs.append(df)
            del df; gc.collect()

    if not all_dfs:
        print("  No data loaded — aborting.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    del all_dfs; gc.collect()

    # Normalize state column
    combined["_state_safe"] = (
        combined["state"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    )

    print(f"  Combined: {len(combined):,} rows × {combined.shape[1]} cols\n")

    # ── Step 3: Write by_state files ─────────────────────────────────────────
    print("  Writing by_state files...\n")

    for state in VALID_STATES:
        safe     = to_safe(state)
        mask     = combined["_state_safe"] == safe
        state_df = combined[mask].drop(columns=["_state_safe"])
        out_path = ML_BY_STATE / f"{safe}.parquet"

        state_df.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {safe}.parquet : {len(state_df):,} rows × {state_df.shape[1]} cols ({size_mb:.0f} MB)")
        del state_df; gc.collect()

    # ── Step 4: Write by_state_by_year files ─────────────────────────────────
    print("\n  Writing by_state_by_year files...\n")

    year_col = "tax_year" if "tax_year" in combined.columns else "year"

    for state in VALID_STATES:
        safe = to_safe(state)
        for year in YEARS:
            mask  = (combined["_state_safe"] == safe) & (combined[year_col] == year)
            chunk = combined[mask].drop(columns=["_state_safe"])

            if chunk.empty:
                print(f"  [EMPTY] {safe}_{year} — no data")
                continue

            out_path = ML_BY_STATE_YEAR / f"{safe}_{year}.parquet"
            chunk.to_parquet(out_path, index=False)
            print(f"  {safe}_{year}.parquet : {len(chunk):,} rows × {chunk.shape[1]} cols")
            del chunk; gc.collect()

    del combined; gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("PHASE 9 — COMPLETE")
    print("=" * 65)

    for label, directory in [
        ("by_year",         ML_BY_YEAR),
        ("by_state",        ML_BY_STATE),
        ("by_state_by_year", ML_BY_STATE_YEAR),
    ]:
        files = sorted(directory.glob("*.parquet"))
        mb    = sum(f.stat().st_size for f in files) / 1e6
        print(f"\n  {label:<20}: {len(files)} files | {mb:.0f} MB total")
        for f in files[:3]:
            print(f"    {f.name}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")

    print("\n✅ Done.")


if __name__ == "__main__":
    run()