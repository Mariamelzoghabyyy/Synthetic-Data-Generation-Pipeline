# 11_split_merged_by_zone_year.py
"""
Split merged_full.parquet into:
  - by_state/state_name.parquet               (all years for that state)
  - by_state_by_year/state_name_year.parquet  (one file per state-year)
  - by_year/year_{y}.parquet                  (all states for that year)

Adds a 'state' string column mapped from the numeric 'zone' column.
"""

import modal
import pandas as pd
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "pandas==2.2.0",
        "pyarrow==15.0.0",
        "numpy==1.26.4",
    ])
    .add_local_file("config.py", "/root/config.py")
    .add_local_file("utils.py",  "/root/utils.py")
)

from config import VOLUME_NAMES

app        = modal.App("taxxx-pipeline-11-split-merged")
merged_vol = modal.Volume.from_name(VOLUME_NAMES["merged"], create_if_missing=True)
logs_vol   = modal.Volume.from_name(VOLUME_NAMES["logs"],   create_if_missing=True)

VOLUMES = {
    "/merged": merged_vol,
    "/logs":   logs_vol,
}

ZONE_STATE_MAP = {
    1: "California",
    2: "Florida",
    3: "Texas",
    4: "New York",
    5: "Illinois"
}

@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=4,
    memory=65_536,
    timeout=3_600,
)
def split_merged():
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    from utils import get_logger, write_parquet

    log = get_logger("11_split_merged", "11_split_merged.log")

    # ── Locate merged_full.parquet ────────────────────────────────────────────
    merged_base = Path("/merged")

    candidates = [
        merged_base / "merged_full.parquet",
        merged_base / "merged" / "merged_full.parquet",
        merged_base / "individuals" / "merged_full.parquet",
    ]
    merged_path = None
    for c in candidates:
        if c.exists():
            merged_path = c
            break

    if merged_path is None:
        found = list(merged_base.rglob("merged_full.parquet"))
        if found:
            merged_path = found[0]
        else:
            raise FileNotFoundError(f"merged_full.parquet not found under {merged_base}")

    log.info("Loading: %s", merged_path)
    df = pd.read_parquet(merged_path)
    
    # ── Map zones to states ───────────────────────────────────────────────────
    log.info("Mapping numeric zones to US states...")
    df["state"] = df["zone"].map(ZONE_STATE_MAP)
    
    log.info(
        "Loaded: %d rows × %d cols  fraud=%.4f",
        len(df), len(df.columns), df["fraud_label"].mean(),
    )

    # ── Output directories ────────────────────────────────────────────────────
    out_state      = merged_base / "split_by_state"
    out_state_year = merged_base / "split_by_state_by_year"
    out_year       = merged_base / "split_by_year"

    for d in [out_state, out_state_year, out_year]:
        d.mkdir(parents=True, exist_ok=True)

    zones = sorted(df["zone"].dropna().unique().astype(int))
    years = sorted(df["tax_year"].dropna().unique().astype(int))

    summary = {
        "total_rows":     len(df),
        "zones":          zones,
        "years":          years,
        "by_state":       {},
        "by_state_year":  {},
        "by_year":        {},
    }

    # ── Split by state ────────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("Writing by_state files...")
    log.info("=" * 50)

    for zone in zones:
        state_name = ZONE_STATE_MAP[zone]
        safe_name  = state_name.lower().replace(" ", "_")
        
        state_df = df[df["zone"] == zone].copy().reset_index(drop=True)
        out      = out_state / f"{safe_name}.parquet"
        write_parquet(state_df, out)

        fraud_rate = float(state_df["fraud_label"].mean())
        summary["by_state"][state_name] = {
            "rows":       len(state_df),
            "fraud_rate": round(fraud_rate, 4),
        }
        log.info("  %s: %8d rows  fraud=%.4f", state_name, len(state_df), fraud_rate)

    # ── Split by state × year ─────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("Writing by_state_by_year files...")
    log.info("=" * 50)

    for zone in zones:
        state_name = ZONE_STATE_MAP[zone]
        safe_name  = state_name.lower().replace(" ", "_")
        state_df   = df[df["zone"] == zone]
        
        for year in years:
            sy_df = state_df[state_df["tax_year"] == year].copy().reset_index(drop=True)
            if len(sy_df) == 0:
                continue
                
            out = out_state_year / f"{safe_name}_{year}.parquet"
            write_parquet(sy_df, out)

            fraud_rate = float(sy_df["fraud_label"].mean())
            key = f"{state_name} {year}"
            summary["by_state_year"][key] = {
                "rows":       len(sy_df),
                "fraud_rate": round(fraud_rate, 4),
            }
            log.info("  %s_%d: %7d rows  fraud=%.4f", safe_name, year, len(sy_df), fraud_rate)

    # ── Split by year ─────────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("Writing by_year files...")
    log.info("=" * 50)

    for year in years:
        year_df = df[df["tax_year"] == year].copy().reset_index(drop=True)
        if len(year_df) == 0:
            continue
            
        out = out_year / f"year_{year}.parquet"
        write_parquet(year_df, out)

        fraud_rate = float(year_df["fraud_label"].mean())
        summary["by_year"][f"year_{year}"] = {
            "rows":       len(year_df),
            "fraud_rate": round(fraud_rate, 4),
        }
        log.info("  year_%d: %8d rows  fraud=%.4f", year, len(year_df), fraud_rate)

    # Save the master file with the new state column
    write_parquet(df, merged_path)
    log.info("Updated merged_full.parquet with 'state' column")

    merged_vol.commit()
    logs_vol.commit()

    return summary


@app.local_entrypoint()
def main():
    result = split_merged.remote()

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)

    print(f"\nTotal rows in merged: {result['total_rows']:,}")

    print("\nBy State:")
    for key, stats in result["by_state"].items():
        print(f"  {key:<12} {stats['rows']:>10,} rows  fraud={stats['fraud_rate']:.4f}")

    print("\nBy State × Year:")
    for key, stats in result["by_state_year"].items():
        print(f"  {key:<18} {stats['rows']:>8,} rows  fraud={stats['fraud_rate']:.4f}")

    print("\nBy Year:")
    for key, stats in result["by_year"].items():
        print(f"  {key:<12} {stats['rows']:>10,} rows  fraud={stats['fraud_rate']:.4f}")