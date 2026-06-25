import pandas as pd
import numpy as np
from config import (
    COLS, YEARS, TARGET_FRAUD_RATES,
    TARGET_ROWS_PER_STATE, LIFECYCLE_BUCKETS,
    LIFECYCLE_TARGET_PROPORTIONS,
    FRAUD_RATE_TOLERANCE
)

# Use 5% tolerance since we are prioritizing whole-person lifecycles over exact row counts
ROW_COUNT_TOLERANCE = 0.05 
LIFECYCLE_TOLERANCE = 0.05

PID   = COLS["person_id"]
YEAR  = COLS["year"]
FRAUD = COLS["fraud_label"]
TTYPE = COLS["taxpayer_type"]

def validate(original: pd.DataFrame, sampled: pd.DataFrame, state: str) -> bool:
    print(f"\n{'='*65}")
    print(f"  VALIDATION: {state.upper()}")
    print(f"{'='*65}")

    all_pass = True

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "✓" if ok else "✗"
        print(f"  {status}  {label:<40} {detail}")
        if not ok:
            all_pass = False

    # 1. Row count (Lenient for Quality Buffer)
    n = len(sampled)
    target = TARGET_ROWS_PER_STATE
    drift = (n - target) / target
    # Pass if it's within -2% to +5%
    is_ok = -0.02 <= drift <= 0.05 
    check("Row count (Target 10k)", is_ok, f"{n:,} rows ({drift:+.1%})")

    # 2. Fraud rate
    actual = sampled[FRAUD].mean()
    tr = TARGET_FRAUD_RATES[state]
    fraud_drift = abs(actual - tr)
    check("Fraud rate", fraud_drift < FRAUD_RATE_TOLERANCE, 
          f"Actual: {actual:.4f} vs Target: {tr:.4f}")

    # 3. Lifecycle Integrity (CRITICAL for TimeGAN)
    # Check if any sampled person has FEWER years than they had in the original data
    orig_counts = original.groupby(PID)[YEAR].nunique()
    samp_counts = sampled.groupby(PID)[YEAR].nunique()
    
    # Align the two series
    common_pids = samp_counts.index
    mismatches = (orig_counts.loc[common_pids] != samp_counts).sum()
    check("Lifecycle integrity (No partial persons)", mismatches == 0, 
          f"{mismatches} broken histories found")

    # 4. Temporal Gap Check (CRITICAL for Logical Consistency)
    # Check if any person has a "gap" in years (e.g., 2020, 2022 but no 2021)
    def has_gap(years):
        years = sorted(list(years))
        if len(years) <= 1: return False
        return max(years) - min(years) + 1 != len(years)

    gaps = sampled.groupby(PID)[YEAR].apply(has_gap).sum()
    check("Temporal consistency (No year gaps)", gaps == 0, 
          f"{gaps} persons have gaps in their timeline")

    # 5. Taxpayer Type Drift
    orig_dist = original[TTYPE].value_counts(normalize=True)
    samp_dist = sampled[TTYPE].value_counts(normalize=True)
    max_tt_drift = 0
    for tt in orig_dist.index:
        drift = abs(orig_dist.get(tt, 0) - samp_dist.get(tt, 0))
        max_tt_drift = max(max_tt_drift, drift)
    check("Taxpayer type distribution drift", max_tt_drift < 0.03, 
          f"Max drift: {max_tt_drift:.2%}")

    # 6. Yearly Distribution (State Character)
    print(f"\n  Yearly Distribution:")
    for yr in YEARS:
        pct = (sampled[YEAR] == yr).mean()
        print(f"    {yr}: {pct:.1%}")

    if all_pass:
        print(f"\n  ✓ {state.upper()} VALIDATION SUCCESSFUL")
    else:
        print(f"\n  ⚠ {state.upper()} VALIDATION WARNINGS FOUND")
        
    return all_pass