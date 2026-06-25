"""
Person-lifecycle-aware stratified sampler.

Core idea
─────────
Every person has a LIFECYCLE TYPE (how many years they filed)
and a FRAUD PATTERN (clean / chronic / mixed).
We sample persons so that the OUTPUT dataset preserves
realistic proportions of all lifecycle × fraud combinations,
then pull ALL rows for each sampled person so no individual
history is broken.

Pipeline per state
──────────────────
1.  Build person index  → one row per person with lifecycle attrs
2.  Classify each person into (lifecycle_bucket × fraud_pattern)
3.  Compute how many persons of each class to sample
    to hit target_rows and target_fraud_rate simultaneously
4.  Stratified sample within each class
    (further stratified by taxpayer_type to preserve that mix)
5.  Pull all rows for sampled persons
6.  Trim excess rows by removing whole persons (clean, smallest first)
    + single-row trim if needed
7.  Validate internally before returning
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    COLS,
    FRAUD_PATTERN_PROPORTIONS,
    LIFECYCLE_BUCKETS,
    LIFECYCLE_TARGET_PROPORTIONS,
    RANDOM_SEED,
    TARGET_FRAUD_RATES,
    TARGET_ROWS_PER_STATE,
)

logger = logging.getLogger(__name__)

# ── Column aliases ─────────────────────────────────────────────────────
PID   = COLS["person_id"]
YEAR  = COLS["year"]
FRAUD = COLS["fraud_label"]
TTYPE = COLS["taxpayer_type"]


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Build person index
# ══════════════════════════════════════════════════════════════════════

def build_person_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the row-level dataframe to one row per person.
    Attaches lifecycle_bucket, fraud_pattern, entry_cohort_band.
    """

    index = (
        df.groupby(PID)
        .agg(
            n_years       = (YEAR,  "nunique"),
            n_rows        = (PID,   "count"),
            fraud_ever    = (FRAUD, "max"),
            fraud_always  = (FRAUD, "min"),
            fraud_mean    = (FRAUD, "mean"),
            fraud_sum     = (FRAUD, "sum"),
            taxpayer_type = (TTYPE, lambda x: x.mode().iloc[0]),
            year_min      = (YEAR,  "min"),
            year_max      = (YEAR,  "max"),
        )
        .reset_index()
    )

    # ── Lifecycle bucket ──────────────────────────────────────────────
    def _lifecycle(n):
        for bucket, (lo, hi) in LIFECYCLE_BUCKETS.items():
            if lo <= n <= hi:
                return bucket
        return "persistent"   # fallback (shouldn't happen)

    index["lifecycle"] = index["n_years"].map(_lifecycle)

    # ── Fraud pattern ─────────────────────────────────────────────────
    # Determines the KIND of fraud trajectory (important for ML)
    conditions = [
        (index["fraud_ever"] == 0),
        (index["fraud_ever"] == 1) & (index["fraud_always"] == 1),
        (index["fraud_ever"] == 1) & (index["fraud_always"] == 0),
    ]
    choices = ["clean", "chronic", "mixed"]
    index["fraud_pattern"] = np.select(conditions, choices, default="clean")

    # For mixed persons: classify direction of fraud trajectory
    # escalator  = fraud appears and stays (0→1 transition, no 1→0)
    # reformed   = fraud disappears and stays gone (1→0, no 0→1)
    # sporadic   = multiple switches
    mixed_mask = index["fraud_pattern"] == "mixed"
    if mixed_mask.any():
        mixed_pids = index.loc[mixed_mask, PID].values
        mixed_df   = df[df[PID].isin(mixed_pids)].copy()
        trajectory = _classify_mixed_trajectories(mixed_df)
        index = index.merge(trajectory, on=PID, how="left")
        # Fill non-mixed with their pattern
        index["fraud_detail"] = index["fraud_detail"].fillna(
            index["fraud_pattern"]
        )
    else:
        index["fraud_detail"] = index["fraud_pattern"]

    # ── Entry cohort band ─────────────────────────────────────────────
    index["cohort_band"] = pd.cut(
        index["year_min"],
        bins=[2018, 2019, 2021, 2025],
        labels=["early", "mid", "late"],
        right=True,
    )

    return index


def _classify_mixed_trajectories(mixed_df: pd.DataFrame) -> pd.DataFrame:
    """
    For persons with mixed fraud: determine if they are
    escalator, reformed, or sporadic.
    Returns DataFrame with columns [person_id, fraud_detail].
    """
    records = []
    for pid, grp in mixed_df.groupby(PID):
        labels = grp.sort_values(YEAR)[FRAUD].tolist()
        transitions = [
            (labels[i], labels[i + 1])
            for i in range(len(labels) - 1)
        ]
        clean_to_fraud = sum(1 for a, b in transitions if a == 0 and b == 1)
        fraud_to_clean = sum(1 for a, b in transitions if a == 1 and b == 0)

        if clean_to_fraud >= 1 and fraud_to_clean == 0:
            detail = "escalator"
        elif fraud_to_clean >= 1 and clean_to_fraud == 0:
            detail = "reformed"
        else:
            detail = "sporadic"

        records.append({PID: pid, "fraud_detail": detail})

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Compute target person counts per stratum
# ══════════════════════════════════════════════════════════════════════

def compute_targets(person_index: pd.DataFrame, state: str) -> Dict[Tuple[str, str, str], int]:
    target_rows = TARGET_ROWS_PER_STATE
    target_rate = TARGET_FRAUD_RATES[state]
    
    # 1. Estimate how many fraud persons we need to hit the row target
    fraud_pool = person_index[person_index["fraud_ever"] == 1]
    avg_rows_fraud = fraud_pool["n_rows"].mean()
    n_fraud_total = round((target_rows * target_rate) / avg_rows_fraud)
    
    # 2. Estimate how many clean persons
    clean_pool = person_index[person_index["fraud_ever"] == 0]
    avg_rows_clean = clean_pool["n_rows"].mean()
    n_clean_total = round(((target_rows * (1 - target_rate)) / avg_rows_clean) * 1.1) # 10% buffer
    
    targets = {}

    # 3. Distribute Fraud into (Bucket x Chronic/Mixed)
    for pattern in ["chronic", "mixed"]:
        pattern_prop = FRAUD_PATTERN_PROPORTIONS["always_fraud" if pattern=="chronic" else "mixed"]
        n_pattern_total = round(n_fraud_total * pattern_prop)
        
        for bucket, bucket_prop in LIFECYCLE_TARGET_PROPORTIONS.items():
            cell = fraud_pool[(fraud_pool["lifecycle"] == bucket) & (fraud_pool["fraud_pattern"] == pattern)]
            n_cell = round(n_pattern_total * bucket_prop)
            targets[(bucket, pattern, "fraud")] = min(n_cell, len(cell))

    # 4. Distribute Clean into (Bucket x Clean)
    for bucket, bucket_prop in LIFECYCLE_TARGET_PROPORTIONS.items():
        cell = clean_pool[clean_pool["lifecycle"] == bucket]
        n_cell = round(n_clean_total * bucket_prop)
        targets[(bucket, "clean", "clean")] = min(n_cell, len(cell))
        
    return targets


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Stratified sampling within each cell
# ══════════════════════════════════════════════════════════════════════

def sample_from_cell(
    pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> List:
    """
    Sample n person_ids from pool.
    Further stratified by taxpayer_type to preserve that distribution.
    Uses largest-remainder allocation to avoid under-sampling.
    """
    if n <= 0:
        return []
    if n >= len(pool):
        return pool[PID].tolist()

    # Stratify by taxpayer_type
    strata   = list(pool.groupby(TTYPE, observed=True))
    total    = len(pool)
    allocs   = []

    for name, grp in strata:
        exact     = n * len(grp) / total
        floor_val = int(exact)
        remainder = exact - floor_val
        allocs.append([name, grp, floor_val, remainder])

    # Distribute leftover slots by largest remainder
    allocated = sum(a[2] for a in allocs)
    shortfall = n - allocated
    allocs.sort(key=lambda x: -x[3])
    for i in range(shortfall):
        allocs[i][2] += 1

    sampled_ids = []
    for _, grp, n_take, _ in allocs:
        n_take = min(n_take, len(grp))
        if n_take > 0:
            chosen = grp.sample(
                n=n_take,
                random_state=int(rng.integers(0, 999_999)),
                replace=False,
            )[PID].tolist()
            sampled_ids.extend(chosen)

    # Safety trim (should rarely trigger)
    if len(sampled_ids) > n:
        rng.shuffle(sampled_ids)
        sampled_ids = sampled_ids[:n]

    return sampled_ids


def collect_all_sampled_ids(person_index: pd.DataFrame, targets: Dict, rng: np.random.Generator) -> List:
    all_ids = []
    for (bucket, pattern, fraud_class), n_target in targets.items():
        if n_target <= 0: continue
        
        # Filter for the specific cell
        cell = person_index[
            (person_index["lifecycle"] == bucket) & 
            (person_index["fraud_pattern"] == pattern)
        ]
        
        ids = sample_from_cell(cell, n_target, rng)
        all_ids.extend(ids)
        logger.info(f"    [{bucket:>10} | {pattern:<8}] target={n_target:>4} got={len(ids):>4}")
    return all_ids


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Pull rows and trim to target
# ══════════════════════════════════════════════════════════════════════

def pull_rows(df: pd.DataFrame, sampled_ids: List) -> pd.DataFrame:
    """Pull all rows for sampled person_ids."""
    return df[df[PID].isin(set(sampled_ids))].copy()


def trim_to_target(
    result: pd.DataFrame,
    person_index: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Quality-First Trimming.
    
    Strategy:
      1. Never break a person's lifecycle (no single-row deletions).
      2. Only remove 'Always Clean' persons to protect fraud/mixed distributions.
      3. Use a soft-target (e.g., 10,000 to 10,250) to minimize data loss.
    """
    target = TARGET_ROWS_PER_STATE
    current = len(result)
    
    # ── 1. Check if we are already in the 'Sweet Spot' ──────────────────
    # If we are within 2.5% of the target, stop. Better to have 10,200 rows 
    # of high-quality data than exactly 10,000 with a skewed distribution.
    if current <= target + (target * 0.025):
        logger.info(f"  Result count {current:,} is within quality buffer. No trimming needed.")
        return result

    excess = current - target
    logger.info(f"  Current count {current:,} exceeds target. Trimming whole clean persons...")

    # ── 2. Identify 'Always Clean' persons from the current sample ──────
    sampled_pids = set(result[PID].unique())
    removable_pool = person_index[
        (person_index[PID].isin(sampled_pids)) & 
        (person_index["fraud_pattern"] == "clean")
    ].copy()

    # ── 3. Randomize the removal pool ───────────────────────────────────
    # We shuffle instead of sorting by size. This ensures we don't 
    # accidentally delete all 'transient' (1-year) or all 'persistent' (7-year) filers.
    removable_pool = removable_pool.sample(
        frac=1, 
        random_state=int(rng.integers(0, 999_999))
    )

    to_remove: set = set()
    freed: int = 0

    for _, row in removable_pool.iterrows():
        # Stop once we get close to the target (allows for that ~10,100 - 10,200 range)
        if (current - freed) <= target + 50:
            break
        to_remove.add(row[PID])
        freed += int(row["n_rows"])

    # ── 4. Apply the trim ───────────────────────────────────────────────
    result = result[~result[PID].isin(to_remove)].copy()
    
    logger.info(
        f"    Removed {len(to_remove):,} whole clean persons "
        f"({freed:,} rows) → {len(result):,} rows remaining."
    )
    
    # NOTE: The "Single-row trim" has been deleted to preserve 
    # logical consistency and temporal dependencies.

    return result


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Internal post-sample report
# ══════════════════════════════════════════════════════════════════════

def log_sample_report(
    result: pd.DataFrame,
    person_index: pd.DataFrame,
    state: str,
) -> None:
    """Log a detailed breakdown of the sampled dataset."""

    sampled_index = person_index[
        person_index[PID].isin(result[PID].unique())
    ]

    logger.info(f"\n  ── Sample report: {state.upper()} ──")
    logger.info(f"  Rows    : {len(result):,}")
    logger.info(f"  Persons : {result[PID].nunique():,}")
    logger.info(f"  Fraud   : {result[FRAUD].mean():.4f}")

    logger.info(f"\n  Lifecycle distribution (persons):")
    lc_counts = sampled_index["lifecycle"].value_counts()
    lc_total  = lc_counts.sum()
    for bucket in LIFECYCLE_BUCKETS:
        cnt  = lc_counts.get(bucket, 0)
        prop = cnt / lc_total if lc_total > 0 else 0
        logger.info(f"    {bucket:<12} : {cnt:>6,}  ({prop:.2%})")

    logger.info(f"\n  Fraud detail distribution (fraud persons):")
    fraud_idx = sampled_index[sampled_index["fraud_ever"] == 1]
    for detail, cnt in fraud_idx["fraud_detail"].value_counts().items():
        logger.info(f"    {detail:<12} : {cnt:>6,}")

    logger.info(f"\n  Rows per year:")
    for yr in sorted(result[YEAR].unique()):
        cnt = (result[YEAR] == yr).sum()
        logger.info(f"    {yr} : {cnt:>7,}")


# ══════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def sample_state(
    state: str,
    df: pd.DataFrame,
    seed_offset: int = 0,
) -> pd.DataFrame:
    """
    Full sampling pipeline for one state.
    df already loaded and cleaned by main.py.
    Returns sampled DataFrame with all original columns intact.
    """
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)

    logger.info(f"\n{'='*65}")
    logger.info(f"  SAMPLING: {state.upper()}")
    logger.info(f"{'='*65}")
    logger.info(
        f"\n  Input: {len(df):,} rows | "
        f"{df[PID].nunique():,} persons | "
        f"fraud={df[FRAUD].mean():.4f}"
    )

    # Step 1
    logger.info("\n  Building person index...")
    person_index = build_person_index(df)

    logger.info("\n  Lifecycle × fraud breakdown (all persons):")
    for (lc, fp), grp in person_index.groupby(
        ["lifecycle", "fraud_pattern"], observed=True
    ):
        logger.info(f"    {lc:<12} | {fp:<8} : {len(grp):>7,} persons")

    # Step 2
    logger.info(f"\n  Computing targets for {state}...")
    targets = compute_targets(person_index, state)

    # Step 3
    logger.info("\n  Sampling persons per cell:")
    sampled_ids = collect_all_sampled_ids(person_index, targets, rng)
    logger.info(f"\n  Total sampled persons: {len(sampled_ids):,}")

    # Step 4
    result = pull_rows(df, sampled_ids)
    logger.info(f"  Rows after pull: {len(result):,}")

    result = trim_to_target(result, person_index, rng)

    # Final shuffle (randomise row order for ML)
    result = result.sample(
        frac=1,
        random_state=RANDOM_SEED + seed_offset
    ).reset_index(drop=True)

    # Step 5
    log_sample_report(result, person_index, state)

    actual_rate  = result[FRAUD].mean()
    target_rate  = TARGET_FRAUD_RATES[state]
    logger.info(
        f"\n  ✓ {state.upper()} DONE  |  "
        f"rows={len(result):,}  |  "
        f"fraud={actual_rate:.4f}  "
        f"(target={target_rate:.4f}  "
        f"drift={actual_rate - target_rate:+.4f})"
    )

    return result