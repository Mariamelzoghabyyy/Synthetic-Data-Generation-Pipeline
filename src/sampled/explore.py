"""
explore.py — run this standalone, no config.py needed.
Just change the paths at the top to match your files.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── CHANGE THESE TO YOUR ACTUAL PATHS ─────────────────────────────────
STATE_FILES = {
    "california": r"D:\projiikkkkttttt\data\by_state\california.parquet",
    "illinois":   r"D:\projiikkkkttttt\data\by_state\illinois.parquet",
    "new_york":   r"D:\projiikkkkttttt\data\by_state\new_york.parquet",
    "florida":    r"D:\projiikkkkttttt\data\by_state\florida.parquet",
    "texas":      r"D:\projiikkkkttttt\data\by_state\texas.parquet",
}

# ── Column names ───────────────────────────────────────────────────────
PID   = "person_id"
YEAR  = "tax_year"
FRAUD = "fraud_label"
FTYPE = "fraud_type"
FCAT  = "fraud_category"
TTYPE = "taxpayer_type"
FYEAR = "first_year_filing"
ECOH  = "entry_cohort"


# ─────────────────────────────────────────────────────────────────────
def explore_state(state: str, path: str) -> dict:

    # ── Check file exists before loading ──────────────────────
    if not Path(path).exists():
        print(f"\n  ✗ FILE NOT FOUND: {path}")
        return {}

    print(f"\n{'='*65}")
    print(f"  {state.upper()}")
    print(f"{'='*65}")
    print(f"  Loading {path} ...")

    df = pd.read_parquet(path)
    print(f"  Done loading.")

    # ── Basic shape ───────────────────────────────────────────
    n_rows    = len(df)
    n_persons = df[PID].nunique()
    print(f"\n  Rows          : {n_rows:,}")
    print(f"  Persons       : {n_persons:,}")
    print(f"  Avg yrs/person: {n_rows / n_persons:.2f}")
    print(f"  Columns       : {len(df.columns)}")

    # ── Fraud rate ────────────────────────────────────────────
    fraud_rate        = df[FRAUD].mean()
    n_fraud_rows      = df[FRAUD].sum()
    n_fraud_persons   = df.groupby(PID)[FRAUD].max().sum()

    print(f"\n  Fraud rate (rows)   : {fraud_rate:.4f}  "
          f"({fraud_rate*100:.2f}%)  —  {int(n_fraud_rows):,} rows")
    print(f"  Fraud persons       : {int(n_fraud_persons):,} / "
          f"{n_persons:,}  "
          f"({n_fraud_persons/n_persons*100:.2f}%)")

    # ── Fraud types ───────────────────────────────────────────
    fraud_rows = df[df[FRAUD] == 1]
    if len(fraud_rows) > 0:
        print(f"\n  Fraud types:")
        for ft, cnt in fraud_rows[FTYPE].value_counts().items():
            print(f"    {ft:<35} {cnt:>8,}")

        print(f"\n  Fraud categories:")
        for fc, cnt in fraud_rows[FCAT].value_counts().items():
            print(f"    {fc:<35} {cnt:>8,}")

    # ── Person filing pattern (how many years per person) ─────
    years_per_person = df.groupby(PID)[YEAR].nunique()
    print(f"\n  Years filed per person:")
    for yrs, cnt in years_per_person.value_counts().sort_index().items():
        bar = "█" * int(cnt / years_per_person.value_counts().max() * 20)
        print(f"    {yrs} yrs : {cnt:>8,}  {bar}")

    # ── Year row distribution ─────────────────────────────────
    print(f"\n  Rows per year:")
    for yr, cnt in df[YEAR].value_counts().sort_index().items():
        print(f"    {yr} : {cnt:>8,}")

    # ── Fraud rate per year ────────────────────────────────────
    print(f"\n  Fraud rate per year:")
    yr_stats = (
        df.groupby(YEAR)[FRAUD]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "rate", "sum": "fraud_rows", "count": "total"})
    )
    for yr, row in yr_stats.iterrows():
        print(f"    {yr} : rate={row['rate']:.4f}  "
              f"fraud={int(row['fraud_rows']):>6,} / {int(row['total']):>8,}")

    # ── Taxpayer type breakdown ────────────────────────────────
    print(f"\n  Taxpayer type + fraud rate:")
    tt_stats = (
        df.groupby(TTYPE)[FRAUD]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "rows"})
        .sort_values("fraud_rate", ascending=False)
    )
    for tt, row in tt_stats.iterrows():
        print(f"    {tt:<25} fraud={row['fraud_rate']:.4f}  "
              f"rows={int(row['rows']):>8,}")

    # ── first_year_filing ─────────────────────────────────────
    print(f"\n  first_year_filing distribution:")
    for val, cnt in df[FYEAR].value_counts(dropna=False).sort_index().items():
        print(f"    {str(val):<6} : {cnt:>8,}")

    # ── entry_cohort ──────────────────────────────────────────
    print(f"\n  entry_cohort distribution:")
    for val, cnt in df[ECOH].value_counts(dropna=False).sort_index().items():
        print(f"    {str(val):<8} : {cnt:>8,}")

    # ── Person fraud consistency ──────────────────────────────
    person_fraud = df.groupby(PID)[FRAUD].agg(["min", "max", "mean"])
    always_clean    = (person_fraud["max"] == 0).sum()
    always_fraud    = (person_fraud["min"] == 1).sum()
    mixed           = (person_fraud["min"] != person_fraud["max"]).sum()

    print(f"\n  Person-level fraud pattern:")
    print(f"    Always clean (0 every year)  : {int(always_clean):>8,}")
    print(f"    Always fraud (1 every year)  : {int(always_fraud):>8,}")
    print(f"    Mixed (escalator/reformed)   : {int(mixed):>8,}")

    if mixed > 0:
        mixed_persons = person_fraud[person_fraud["min"] != person_fraud["max"]]
        print(f"    Mixed fraud rate distribution:")
        print(f"      mean  = {mixed_persons['mean'].mean():.4f}")
        print(f"      min   = {mixed_persons['mean'].min():.4f}")
        print(f"      max   = {mixed_persons['mean'].max():.4f}")

    print(f"\n  ✓ {state.upper()} complete")

    return {
        "state":              state,
        "n_rows":             n_rows,
        "n_persons":          n_persons,
        "fraud_rate":         fraud_rate,
        "fraud_persons":      int(n_fraud_persons),
        "mixed_fraud":        int(mixed),
        "always_fraud":       int(always_fraud),
        "always_clean":       int(always_clean),
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  TAX EVASION DATA EXPLORER")
print("=" * 65)

results = {}

for state, path in STATE_FILES.items():
    try:
        results[state] = explore_state(state, path)
    except Exception as e:
        print(f"\n  ✗ ERROR on {state}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# ── Cross-state summary ───────────────────────────────────────────────
if results:
    print("\n" + "█" * 65)
    print("  CROSS-STATE SUMMARY")
    print("█" * 65)
    print(
        f"\n{'State':<14} {'Rows':>10} {'Persons':>10} "
        f"{'Fraud%':>8} {'AvgYrs':>8} "
        f"{'AlwaysFraud':>12} {'Mixed':>8}"
    )
    print("-" * 65)

    for r in results.values():
        if not r:
            continue
        avg_yrs = r["n_rows"] / r["n_persons"]
        print(
            f"{r['state']:<14} {r['n_rows']:>10,} {r['n_persons']:>10,} "
            f"{r['fraud_rate']*100:>7.2f}% {avg_yrs:>8.2f} "
            f"{r['always_fraud']:>12,} {r['mixed_fraud']:>8,}"
        )

    print("\n  Done. Use the fraud rates above to update STATE_FRAUD_RATES in config.py")
else:
    print("\n  No results — check your file paths above")