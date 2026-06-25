# 06_generate_splits.py
"""
Assemble train / val / test splits from panel parquets.
Writes seed_panels/ files for GAN input.

Split strategy:
  Use Case A — Known-taxpayer detection (year-level split):
    Train: 2019-2023 all persons
    Val:   2024 all persons (includes persons seen in train)
    Test:  2025 all persons (includes persons seen in training)

  Use Case B — New-taxpayer detection (person-level split):
    Train: 2019-2023 all persons + 2024/2025 rows for persons
           already seen in training years
    Val:   2024 rows for persons NOT seen in 2019-2023
    Test:  2025 rows for persons NOT seen in 2019-2023

Both splits are written. Metrics logged for each.

Outputs (all paths from config):
  Use Case A:
    TRAIN_ALL, TRAIN_EMPLOYEES, TRAIN_SE, TRAIN_ITEMIZERS
    VAL_2024, TEST_2025

  Use Case B (subfolder: person_level/):
    train_person_level.parquet
    val_person_level.parquet
    test_person_level.parquet

  GAN seeds (always from Use Case A training data):
    SEED_W2, SEED_SE, SEED_ITEMIZERS
    SEED_COMPLIANT, SEED_EVADERS
"""

import modal
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# ── Image ─────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "pandas==2.2.0",
        "numpy==1.26.4",
        "pyarrow==15.0.0",
    ])
    .add_local_file("config.py", "/root/config.py")
    .add_local_file("utils.py",  "/root/utils.py")
    .add_local_file("feature_config.py", "/root/feature_config.py")  # add this
)

# ── App & volumes ─────────────────────────────────────────────────────────────
from config import VOLUME_NAMES

app        = modal.App("taxxx-pipeline-06-splits")
final_vol  = modal.Volume.from_name(VOLUME_NAMES["final"],  create_if_missing=True)
panels_vol = modal.Volume.from_name(VOLUME_NAMES["panels"], create_if_missing=True)
logs_vol   = modal.Volume.from_name(VOLUME_NAMES["logs"],   create_if_missing=True)

VOLUMES = {
    "/final_dataset": final_vol,
    "/seed_panels":   panels_vol,
    "/logs":          logs_vol,
}

# ── Split constants ───────────────────────────────────────────────────────────
TRAIN_YEARS = list(range(2019, 2024))   # 2019-2023 inclusive
VAL_YEAR    = 2024
TEST_YEAR   = 2025

# Target GAN seed size — large enough to learn distribution,
# small enough for reasonable training time on T4
GAN_SEED_TARGET = 50_000

# Taxpayer type groupings
W2_TYPES   = frozenset(["pure_w2", "w2_with_side_biz"])
SE_TYPES   = frozenset(["pure_se", "gig_only", "business_owner", "multi_biz_owner"])


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions — no Modal context needed
# ─────────────────────────────────────────────────────────────────────────────

def _fix_contradictions(df: pd.DataFrame, name: str, log) -> pd.DataFrame:
    """Flip fraud_label=1/fraud_type=none rows back to compliant."""
    mask = (df["fraud_label"] == 1) & (df["fraud_type"] == "none")
    n    = mask.sum()
    if n > 0:
        log.info("%s: fixing %d fraud_label=1/fraud_type=none rows → fraud_label=0", name, n)
        df.loc[mask, "fraud_label"]        = 0
        df.loc[mask, "fraud_category"]     = "none"
        df.loc[mask, "evasion_amount"]     = None
        df.loc[mask, "evasion_rate"]       = None
        df.loc[mask, "tax_gap_amount"]     = None
        df.loc[mask, "true_tax_liability"] = None
    return df


def _log_split_metrics(
    df:             pd.DataFrame,
    name:           str,
    fraud_target:   float,
    fraud_tol:      float,
    log,
) -> None:
    """
    Log a full diagnostic block for one split.
    Covers: row count, fraud rate, taxpayer mix,
    zone mix, year mix, known/new person breakdown.
    """
    if len(df) == 0:
        log.warning("%s: EMPTY — no rows", name)
        return

    fraud_actual = float(df["fraud_label"].mean())
    fraud_ok     = abs(fraud_actual - fraud_target) <= fraud_tol
    fraud_status = "✓" if fraud_ok else "✗"

    log.info("─" * 60)
    log.info(
        "%s %s: %d rows  fraud=%.4f  (target=%.4f ±%.3f)",
        fraud_status, name, len(df),
        fraud_actual, fraud_target, fraud_tol,
    )

    # Taxpayer type distribution
    type_dist = df["taxpayer_type"].value_counts(normalize=True).round(3)
    log.info("  Taxpayer type distribution:\n%s", type_dist.to_string())

    # Zone distribution
    zone_dist = (
        df["zone"].value_counts(normalize=True)
          .sort_index().round(3)
    )
    log.info("  Zone distribution:\n%s", zone_dist.to_string())

    # Year distribution (only meaningful if multiple years present)
    if "tax_year" in df.columns and df["tax_year"].nunique() > 1:
        year_stats = df.groupby("tax_year").agg(
            rows=("fraud_label", "count"),
            fraud=("fraud_label", "mean"),
        ).round(4)
        log.info("  By year:\n%s", year_stats.to_string())

    # Fraud type distribution among evaders
    evaders = df[df["fraud_label"] == 1]
    if len(evaders) > 0 and "fraud_type" in df.columns:
        ftype_dist = evaders["fraud_type"].value_counts(normalize=True).round(3)
        log.info("  Fraud types (evaders only):\n%s", ftype_dist.to_string())

    # Income sanity
    for col in ["agi", "total_tax_liability", "w2_wages"]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                log.info(
                    "  %-25s  median=%10,.0f  mean=%10,.0f  pct_zero=%.1f%%",
                    col,
                    float(s.median()),
                    float(s.mean()),
                    float((s == 0).mean() * 100),
                )


def _known_vs_new_breakdown(
    df:          pd.DataFrame,
    train_pids:  set,
    name:        str,
    log,
) -> None:
    """
    Log fraud rate and row count separately for:
      - Persons whose person_id appears in the training set (known)
      - Persons whose person_id does NOT appear in training (new)
    This is the key metric for understanding leakage risk.
    """
    known_mask = df["person_id"].isin(train_pids)
    known      = df[known_mask]
    new        = df[~known_mask]

    log.info(
        "  %s | Known persons: %d rows  fraud=%.4f",
        name,
        len(known),
        float(known["fraud_label"].mean()) if len(known) else 0.0,
    )
    log.info(
        "  %s | New persons:   %d rows  fraud=%.4f",
        name,
        len(new),
        float(new["fraud_label"].mean()) if len(new) else 0.0,
    )
    if len(df) > 0:
        log.info(
            "  %s | New-person share: %.1f%%",
            name,
            len(new) / len(df) * 100,
        )


def _stratified_sample(
    df:       pd.DataFrame,
    n:        int,
    stratify: str,
    rng:      np.random.Generator,
) -> pd.DataFrame:
    """
    Sample exactly n rows stratified by `stratify` column.
    Each group gets floor(n * group_share) rows.
    Any remainder is filled by random sampling across groups.
    If df has fewer than n rows, returns df unchanged.
    """
    if len(df) <= n:
        return df.copy()

    group_counts = df[stratify].value_counts()
    total        = len(df)
    alloc        = (group_counts / total * n).astype(int)

    # Remainder allocation — give extra rows to largest groups first
    remainder = n - alloc.sum()
    for grp in alloc.index[:remainder]:
        alloc[grp] += 1

    parts: list[pd.DataFrame] = []
    for grp, cnt in alloc.items():
        grp_df = df[df[stratify] == grp]
        take   = min(cnt, len(grp_df))
        if take > 0:
            parts.append(
                grp_df.sample(
                    n=take,
                    random_state=int(rng.integers(0, 2**31)),
                )
            )

    return pd.concat(parts, ignore_index=True)


def _enforce_fraud_rate(
    df:         pd.DataFrame,
    target:     float,
    tolerance:  float,
    rng:        np.random.Generator,
    log,
    name:       str = "",
) -> pd.DataFrame:
    """
    Enforce fraud rate on df by upsampling the minority class
    if the actual rate falls outside [target - tolerance, target + tolerance].

    Strategy:
      - If fraud rate too LOW:  oversample existing evader rows with replacement
      - If fraud rate too HIGH: oversample existing compliant rows with replacement

    Does NOT drop any rows — only adds rows to bring rate into band.
    The added rows are near-duplicates with small label noise — acceptable
    for a training set but the raw panel is still preserved.
    """
    actual = float(df["fraud_label"].mean())
    lo     = target - tolerance
    hi     = target + tolerance

    if lo <= actual <= hi:
        log.info(
            "  %s fraud rate %.4f already in [%.4f, %.4f] — no adjustment",
            name, actual, lo, hi,
        )
        return df

    log.info(
        "  %s fraud rate %.4f outside [%.4f, %.4f] — adjusting...",
        name, actual, lo, hi,
    )

    evaders   = df[df["fraud_label"] == 1]
    compliant = df[df["fraud_label"] == 0]
    n_total   = len(df)

    if actual < lo:
        # Need more evaders
        # Target: evaders / (evaders + compliant) = target
        # → n_evaders_needed = target * n_compliant / (1 - target)
        n_evaders_needed = int(np.ceil(
            target * len(compliant) / (1.0 - target)
        ))
        n_to_add = n_evaders_needed - len(evaders)
        if n_to_add > 0 and len(evaders) > 0:
            extra = evaders.sample(
                n=n_to_add,
                replace=True,
                random_state=int(rng.integers(0, 2**31)),
            )
            df = pd.concat([df, extra], ignore_index=True)

    else:
        # Need more compliant rows
        n_compliant_needed = int(np.ceil(
            (1.0 - target) * len(evaders) / target
        ))
        n_to_add = n_compliant_needed - len(compliant)
        if n_to_add > 0 and len(compliant) > 0:
            extra = compliant.sample(
                n=n_to_add,
                replace=True,
                random_state=int(rng.integers(0, 2**31)),
            )
            df = pd.concat([df, extra], ignore_index=True)

    final_rate = float(df["fraud_label"].mean())
    log.info(
        "  %s after adjustment: %d rows  fraud=%.4f",
        name, len(df), final_rate,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Modal function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=4,
    memory=32_768,    # bumped — we hold multiple large DFs in memory
    timeout=7_200,
)
def run_splits():
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    from config import (
        PANEL_BY_DIR,
        TRAIN_DIR, TRAIN_ALL, TRAIN_EMPLOYEES, TRAIN_SE, TRAIN_ITEMIZERS,
        VAL_DIR,   VAL_2024,
        TEST_DIR,  TEST_2025,
        SEED_DIR,  SEED_W2, SEED_SE, SEED_ITEMIZERS,
        SEED_COMPLIANT, SEED_EVADERS,
        FINAL_BASE,
        RANDOM_SEED,
        FRAUD_RATE_OVERALL, FRAUD_RATE_TOLERANCE,
        FRAUD_RATE_W2, FRAUD_RATE_SE,
    )
    from utils import get_logger, write_parquet

    log = get_logger("06_splits", "06_splits.log")
    rng = np.random.default_rng(RANDOM_SEED + 6)

    # ── Output directories ────────────────────────────────────────────────────
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, SEED_DIR,
              FINAL_BASE / "individuals" / "person_level"]:
        d.mkdir(parents=True, exist_ok=True)

    PERSON_LEVEL_DIR = FINAL_BASE / "individuals" / "person_level"

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1 — Load all years
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STEP 1: Loading panel years")
    log.info("=" * 60)

    def _load_year(year: int) -> Optional[pd.DataFrame]:
        p = PANEL_BY_DIR / f"all_zones_{year}.parquet"
        if not p.exists():
            log.warning("Missing year file: %s", p)
            return None
        df = pd.read_parquet(p)
        log.info(
            "  Loaded %d: %d rows  fraud=%.4f",
            year, len(df), float(df["fraud_label"].mean()),
        )
        return df

    # Training years
    train_dfs = [_load_year(y) for y in TRAIN_YEARS]
    train_dfs = [d for d in train_dfs if d is not None]

    if not train_dfs:
        raise FileNotFoundError(
            f"No training year files found in {PANEL_BY_DIR}. "
            "Run 05_generate_panels.py first."
        )

    # Val and test years
    val_raw  = _load_year(VAL_YEAR)
    test_raw = _load_year(TEST_YEAR)

    if val_raw is None:
        raise FileNotFoundError(
            f"Val file all_zones_{VAL_YEAR}.parquet not found in {PANEL_BY_DIR}"
        )
    if test_raw is None:
        raise FileNotFoundError(
            f"Test file all_zones_{TEST_YEAR}.parquet not found in {PANEL_BY_DIR}"
        )

    # Raw training concatenation — no enforcement yet
    train_raw = pd.concat(train_dfs, ignore_index=True)
    log.info(
        "Raw train (all years): %d rows  fraud=%.4f",
        len(train_raw), float(train_raw["fraud_label"].mean()),
    )

    # ── Fix contradictions BEFORE any processing ─────────────────────────────
    log.info("=" * 60)
    log.info("Fixing fraud_label=1 / fraud_type=none contradictions")
    log.info("=" * 60)
    train_raw = _fix_contradictions(train_raw, "TRAIN", log)
    val_raw   = _fix_contradictions(val_raw,   "VAL",   log)
    test_raw  = _fix_contradictions(test_raw,  "TEST",  log)

    # Person ID sets — computed AFTER fixing contradictions
    train_pids = set(train_raw["person_id"].unique())
    val_pids   = set(val_raw["person_id"].unique())
    test_pids  = set(test_raw["person_id"].unique())

    log.info(
        "Unique persons — train: %d  val: %d  test: %d",
        len(train_pids), len(val_pids), len(test_pids),
    )
    log.info(
        "Person overlap — val∩train: %d  test∩train: %d  val∩test: %d",
        len(val_pids  & train_pids),
        len(test_pids & train_pids),
        len(val_pids  & test_pids),
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2 — USE CASE A: Year-level split
    # Known-taxpayer detection — val/test include persons seen in training
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STEP 2: Use Case A — Year-level split")
    log.info("=" * 60)

    # Enforce fraud rate on training set
    train_a = _enforce_fraud_rate(
        train_raw,
        target=FRAUD_RATE_OVERALL,
        tolerance=FRAUD_RATE_TOLERANCE,
        rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        log=log,
        name="TRAIN_A",
    )

    # Derive subsets AFTER enforcement
    train_a_emp  = train_a[train_a["taxpayer_type"].isin(W2_TYPES)].copy()
    train_a_se   = train_a[train_a["taxpayer_type"].isin(SE_TYPES)].copy()
    train_a_item = (
        train_a[train_a["uses_itemized"] == 1].copy()
        if "uses_itemized" in train_a.columns
        else None
    )

    # Val and test use raw panel — no enforcement
    # (real IRS data would not be resampled at test time)
    val_a  = val_raw.copy()
    test_a = test_raw.copy()

    # Write Use Case A splits
    write_parquet(train_a,      TRAIN_ALL)
    write_parquet(train_a_emp,  TRAIN_EMPLOYEES)
    write_parquet(train_a_se,   TRAIN_SE)

    if train_a_item is not None:
        write_parquet(train_a_item, TRAIN_ITEMIZERS)

    write_parquet(val_a,  VAL_2024)
    write_parquet(test_a, TEST_2025)

    # Metrics
    _log_split_metrics(
        train_a, "USE_CASE_A TRAIN_ALL",
        FRAUD_RATE_OVERALL, FRAUD_RATE_TOLERANCE, log,
    )
    _log_split_metrics(
        train_a_emp, "USE_CASE_A TRAIN_EMPLOYEES",
        FRAUD_RATE_W2, FRAUD_RATE_TOLERANCE, log,
    )
    _log_split_metrics(
        train_a_se, "USE_CASE_A TRAIN_SE",
        FRAUD_RATE_SE, FRAUD_RATE_TOLERANCE, log,
    )
    _log_split_metrics(
        val_a, "USE_CASE_A VAL_2024",
        FRAUD_RATE_OVERALL, FRAUD_RATE_TOLERANCE, log,
    )
    _log_split_metrics(
        test_a, "USE_CASE_A TEST_2025",
        FRAUD_RATE_OVERALL, FRAUD_RATE_TOLERANCE, log,
    )

    # Known vs new breakdown for val and test
    log.info("Known vs New person breakdown (Use Case A):")
    _known_vs_new_breakdown(val_a,  train_pids, "VAL_2024",  log)
    _known_vs_new_breakdown(test_a, train_pids, "TEST_2025", log)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — USE CASE B: Person-level split
    # New-taxpayer detection — val/test contain ONLY unseen persons
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STEP 3: Use Case B — Person-level split")
    log.info("=" * 60)

    # Val B: only persons NOT seen in training years
    val_b_new     = val_raw[~val_raw["person_id"].isin(train_pids)].copy()
    val_b_leaked  = val_raw[val_raw["person_id"].isin(train_pids)].copy()

    # Test B: only persons NOT seen in training years
    test_b_new    = test_raw[~test_raw["person_id"].isin(train_pids)].copy()
    test_b_leaked = test_raw[test_raw["person_id"].isin(train_pids)].copy()

    log.info(
        "Val  2024 — new: %d rows  leaked: %d rows",
        len(val_b_new), len(val_b_leaked),
    )
    log.info(
        "Test 2025 — new: %d rows  leaked: %d rows",
        len(test_b_new), len(test_b_leaked),
    )

    # Train B: raw training years + leaked val/test rows
    # (persons from 2024/2025 who were already in training get moved to train)
    train_b = pd.concat(
        [train_raw, val_b_leaked, test_b_leaked],
        ignore_index=True,
    )

    # Enforce fraud rate on training set
    train_b = _enforce_fraud_rate(
        train_b,
        target=FRAUD_RATE_OVERALL,
        tolerance=FRAUD_RATE_TOLERANCE,
        rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        log=log,
        name="TRAIN_B",
    )

    # Write Use Case B splits
    write_parquet(
        train_b,
        PERSON_LEVEL_DIR / "train_person_level.parquet",
    )
    write_parquet(
        val_b_new,
        PERSON_LEVEL_DIR / "val_person_level.parquet",
    )
    write_parquet(
        test_b_new,
        PERSON_LEVEL_DIR / "test_person_level.parquet",
    )

    # Metrics
    _log_split_metrics(
        train_b, "USE_CASE_B TRAIN",
        FRAUD_RATE_OVERALL, FRAUD_RATE_TOLERANCE, log,
    )
    _log_split_metrics(
        val_b_new, "USE_CASE_B VAL_2024 (new persons only)",
        FRAUD_RATE_OVERALL, FRAUD_RATE_TOLERANCE, log,
    )
    _log_split_metrics(
        test_b_new, "USE_CASE_B TEST_2025 (new persons only)",
        FRAUD_RATE_OVERALL, FRAUD_RATE_TOLERANCE, log,
    )

    # Warn if Use Case B val/test are very small
    # (this is expected — only entry_cohort==2024 and 2025 persons)
    for split_name, split_df in [
        ("VAL_B",  val_b_new),
        ("TEST_B", test_b_new),
    ]:
        if len(split_df) < 5_000:
            log.warning(
                "%s has only %d rows — may be too small for reliable evaluation. "
                "Consider using Use Case A metrics as primary.",
                split_name, len(split_df),
            )
        if len(split_df) > 0:
            fraud_b = float(split_df["fraud_label"].mean())
            if abs(fraud_b - FRAUD_RATE_OVERALL) > 0.05:
                log.warning(
                    "%s fraud rate %.4f deviates >5pp from target %.4f — "
                    "new-person cohort may have different fraud characteristics.",
                    split_name, fraud_b, FRAUD_RATE_OVERALL,
                )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4 — GAN seed panels
    # Always derived from Use Case A training data
    # Stratified samples of GAN_SEED_TARGET rows each
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STEP 4: Writing GAN seed panels")
    log.info("=" * 60)

    # W2 seed
    seed_w2 = _stratified_sample(
        train_a_emp,
        n=GAN_SEED_TARGET,
        stratify="taxpayer_type",
        rng=np.random.default_rng(int(rng.integers(0, 2**31))),
    )
    write_parquet(seed_w2, SEED_W2)
    log.info(
        "seed_w2:        %d rows  fraud=%.4f -> %s",
        len(seed_w2), float(seed_w2["fraud_label"].mean()), SEED_W2,
    )

    # SE seed
    seed_se = _stratified_sample(
        train_a_se,
        n=GAN_SEED_TARGET,
        stratify="taxpayer_type",
        rng=np.random.default_rng(int(rng.integers(0, 2**31))),
    )
    write_parquet(seed_se, SEED_SE)
    log.info(
        "seed_se:        %d rows  fraud=%.4f -> %s",
        len(seed_se), float(seed_se["fraud_label"].mean()), SEED_SE,
    )

    # Itemizer seed
    if train_a_item is not None and len(train_a_item) > 0:
        seed_item = _stratified_sample(
            train_a_item,
            n=GAN_SEED_TARGET,
            stratify="taxpayer_type",
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        )
        write_parquet(seed_item, SEED_ITEMIZERS)
        log.info(
            "seed_itemizers: %d rows  fraud=%.4f -> %s",
            len(seed_item), float(seed_item["fraud_label"].mean()), SEED_ITEMIZERS,
        )
    else:
        log.warning("No itemizer rows found — skipping SEED_ITEMIZERS")

    # TimeGAN seeds — balanced compliant/evader
    # Use all evaders from training, sample compliant at 4:1 ratio
    evaders_seed   = train_a[train_a["fraud_label"] == 1].copy()
    compliant_pool = train_a[train_a["fraud_label"] == 0]

    n_evaders        = len(evaders_seed)
    n_compliant_want = min(n_evaders * 4, len(compliant_pool))

    compliant_seed = compliant_pool.sample(
        n=n_compliant_want,
        random_state=int(rng.integers(0, 2**31)),
    )

    write_parquet(compliant_seed, SEED_COMPLIANT)
    write_parquet(evaders_seed,   SEED_EVADERS)

    log.info(
        "seed_compliant: %d rows -> %s",
        len(compliant_seed), SEED_COMPLIANT,
    )
    log.info(
        "seed_evaders:   %d rows -> %s",
        len(evaders_seed), SEED_EVADERS,
    )
    log.info(
        "TimeGAN class ratio: %.1f:1 (compliant:evader)",
        len(compliant_seed) / max(n_evaders, 1),
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5 — Final summary
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STEP 5: Final summary")
    log.info("=" * 60)

    summary = {
        # Use Case A
        "a_train_rows":      len(train_a),
        "a_train_fraud":     round(float(train_a["fraud_label"].mean()), 4),
        "a_val_rows":        len(val_a),
        "a_val_fraud":       round(float(val_a["fraud_label"].mean()), 4),
        "a_test_rows":       len(test_a),
        "a_test_fraud":      round(float(test_a["fraud_label"].mean()), 4),
        # Use Case B
        "b_train_rows":      len(train_b),
        "b_train_fraud":     round(float(train_b["fraud_label"].mean()), 4),
        "b_val_rows":        len(val_b_new),
        "b_val_fraud":       round(float(val_b_new["fraud_label"].mean()), 4) if len(val_b_new) else 0.0,
        "b_test_rows":       len(test_b_new),
        "b_test_fraud":      round(float(test_b_new["fraud_label"].mean()), 4) if len(test_b_new) else 0.0,
        # GAN seeds
        "seed_w2_rows":      len(seed_w2),
        "seed_se_rows":      len(seed_se),
        "seed_evader_rows":  len(evaders_seed),
        "status":            "ok",
    }

    log.info("Use Case A (year-level):")
    log.info(
        "  Train %d rows (fraud=%.4f) | Val %d rows (fraud=%.4f) | Test %d rows (fraud=%.4f)",
        summary["a_train_rows"], summary["a_train_fraud"],
        summary["a_val_rows"],   summary["a_val_fraud"],
        summary["a_test_rows"],  summary["a_test_fraud"],
    )
    log.info("Use Case B (person-level):")
    log.info(
        "  Train %d rows (fraud=%.4f) | Val %d rows (fraud=%.4f) | Test %d rows (fraud=%.4f)",
        summary["b_train_rows"], summary["b_train_fraud"],
        summary["b_val_rows"],   summary["b_val_fraud"],
        summary["b_test_rows"],  summary["b_test_fraud"],
    )

    final_vol.commit()
    panels_vol.commit()
    logs_vol.commit()

    log.info("06_generate_splits complete.")
    return summary


@app.local_entrypoint()
def main():
    result = run_splits.remote()

    print("\n" + "=" * 60)
    print("SPLIT RESULTS")
    print("=" * 60)
    print("\nUse Case A — Year-level (known-taxpayer detection):")
    print(f"  Train: {result['a_train_rows']:>10,} rows  fraud={result['a_train_fraud']:.4f}")
    print(f"  Val:   {result['a_val_rows']:>10,} rows  fraud={result['a_val_fraud']:.4f}")
    print(f"  Test:  {result['a_test_rows']:>10,} rows  fraud={result['a_test_fraud']:.4f}")

    print("\nUse Case B — Person-level (new-taxpayer detection):")
    print(f"  Train: {result['b_train_rows']:>10,} rows  fraud={result['b_train_fraud']:.4f}")
    print(f"  Val:   {result['b_val_rows']:>10,} rows  fraud={result['b_val_fraud']:.4f}")
    print(f"  Test:  {result['b_test_rows']:>10,} rows  fraud={result['b_test_fraud']:.4f}")

    print("\nGAN Seeds:")
    print(f"  W2:      {result['seed_w2_rows']:>8,} rows")
    print(f"  SE:      {result['seed_se_rows']:>8,} rows")
    print(f"  Evaders: {result['seed_evader_rows']:>8,} rows")
    print(f"\nStatus: {result['status']}")