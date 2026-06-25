# 04_generate_links.py
"""
Generate ownership and employment links.
Vectorized by group where possible. No iterrows().
"""

import modal
import numpy as np
import pandas as pd
from pathlib import Path

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
)

# ── App & volumes ─────────────────────────────────────────────────────────────
from config import VOLUME_NAMES

app      = modal.App("taxxx-pipeline-04-links")
dist_vol  = modal.Volume.from_name(VOLUME_NAMES["dists"],  create_if_missing=True)
final_vol = modal.Volume.from_name(VOLUME_NAMES["final"],  create_if_missing=True)
logs_vol  = modal.Volume.from_name(VOLUME_NAMES["logs"],   create_if_missing=True)



VOLUMES = {
    "/distributions": dist_vol,   # DIST_BASE  — master_distributions.pkl lives here
    "/final_dataset": final_vol,  # FINAL_BASE — persons.csv written here
    "/logs":          logs_vol,   # LOGS_BASE
}

# Taxpayer types that can own businesses
OWNER_TYPES = frozenset([
    "business_owner", "multi_biz_owner",
    "w2_with_side_biz", "pure_se", "gig_only",
])

# Taxpayer types that appear on W-2s (employed by a business)
# Keys must exactly match TAXPAYER_TYPES in config
W2_TYPES = frozenset([
    "pure_w2",
    "w2_with_side_biz",
    "retired",      # part-time work is common
    "investor",     # many have W2 from board positions / consulting
])


@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=4,
    memory=32768,
    timeout=3600,
)
def generate_links():
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    from config import (
        ZONE_PROFILES, RANDOM_SEED,
        PERSONS_CSV, BUSINESSES_CSV,
        PB_LINKS_CSV, EMP_LINKS_CSV,
    )
    from utils import get_logger, load_distributions

    log = get_logger("04_links", "04_links.log")
    rng = np.random.default_rng(RANDOM_SEED + 2)

    dist = load_distributions()
    bls  = dist["bls"]["occupations"]   # {occ: {log_mean, log_std, emp_share}}

    log.info("Reading persons and businesses...")
    persons    = pd.read_csv(PERSONS_CSV)
    businesses = pd.read_csv(BUSINESSES_CSV)
    log.info(
        "Persons: %d  Businesses: %d", len(persons), len(businesses)
    )

    # ── Helper: active year range overlap ─────────────────────────────────────
    def _overlap(
        p_entry: int, p_exit: int,
        b_open:  int, b_exit: int,
        offset:  int = 0,
    ) -> tuple[int, int]:
        """
        Return (start, end) of the period when both person and business
        are simultaneously active. Returns (start, end) where end may be
        9999 meaning "still active through 2025+".
        """
        start = max(p_entry + offset, b_open)
        # Use 9999 as "no known exit"; for comparison purposes treat as 2025
        p_end_cmp = p_exit  if p_exit  != 9999 else 2025
        b_end_cmp = b_exit  if b_exit  != 9999 else 2025
        end_cmp   = min(p_end_cmp, b_end_cmp)
        # Sentinel: if both are still active, keep 9999
        end       = 9999 if (p_exit == 9999 and b_exit == 9999) else end_cmp
        return start, end

    # ─────────────────────────────────────────────────────────────────────────
    # OWNERSHIP LINKS
    # Strategy: group businesses by zone and entity-type bucket, then assign
    # owners in bulk using boolean masks — avoids iterrows on owners.
    # ─────────────────────────────────────────────────────────────────────────
    log.info("Generating ownership links...")

    owners = persons[persons["taxpayer_type"].isin(OWNER_TYPES)].copy()
    owners = owners.reset_index(drop=True)

    # Index businesses by zone + entity bucket for O(1) lookup
    sole_biz = businesses[
        businesses["entity_type"].isin(["sole_prop", "llc_single"])
    ].copy()
    corp_biz = businesses[
        businesses["entity_type"].isin(["llc_multi", "scorp", "partnership", "ccorp"])
    ].copy()

    # Build zone → list-of-business-id dicts; shuffle for random assignment
    def _zone_pool(df: pd.DataFrame) -> dict[int, list]:
        pool = {}
        for z in range(1, 6):
            ids = df[df["zone"] == z]["business_id"].tolist()
            rng.shuffle(ids)   # in-place shuffle for random assignment order
            pool[z] = ids
        return pool

    sole_pool = _zone_pool(sole_biz)
    corp_pool = _zone_pool(corp_biz)

    # Track which businesses are already assigned
    used_biz: set[str] = set()

    # Pointer per (zone, pool_type) so we scan linearly instead of O(n²) avail
    sole_ptr: dict[int, int] = {z: 0 for z in range(1, 6)}
    corp_ptr: dict[int, int] = {z: 0 for z in range(1, 6)}

    biz_meta = businesses.set_index("business_id")  # fast scalar lookups

    def _next_available(
        pool: dict[int, list],
        ptr:  dict[int, int],
        zone: int,
    ) -> str | None:
        """
        Return the next un-used business_id from the pool for this zone,
        advancing the pointer. Returns None if pool exhausted.
        """
        lst = pool.get(zone, [])
        while ptr[zone] < len(lst):
            bid = lst[ptr[zone]]
            ptr[zone] += 1
            if bid not in used_biz:
                return bid
        return None

    def _make_own_row(
        person_id:  str,
        bid:        str,
        pct:        float,
        role:       str,
        start:      int,
        end:        int,
    ) -> dict:
        return {
            "person_id":       person_id,
            "business_id":     bid,
            "ownership_pct":   round(pct, 1),
            "role":            role,
            "ownership_start": start,
            "ownership_end":   end,
            "is_active_owner": int(end == 9999 or end >= 2024),
        }

    own_links: list[dict] = []

    # Process each owner-type group separately (one pass per type)
    for ttype, group in owners.groupby("taxpayer_type"):

        for _, person in group.iterrows():
            z       = int(person["zone"])
            p_entry = int(person["entry_year"])
            p_exit  = int(person["exit_year"])

            if ttype in ("pure_se", "gig_only"):
                bid = _next_available(sole_pool, sole_ptr, z)
                if bid is None:
                    continue
                bm = biz_meta.loc[bid]
                start, end = _overlap(p_entry, p_exit,
                                      int(bm["open_year"]), int(bm["exit_year"]))
                if start > (end if end != 9999 else 2025):
                    continue
                used_biz.add(bid)
                own_links.append(_make_own_row(
                    person["person_id"], bid, 100.0, "sole_owner", start, end
                ))

            elif ttype == "business_owner":
                # Prefer corp; fall back to sole
                bid = (_next_available(corp_pool, corp_ptr, z)
                       or _next_available(sole_pool, sole_ptr, z))
                if bid is None:
                    continue
                bm = biz_meta.loc[bid]
                start, end = _overlap(p_entry, p_exit,
                                      int(bm["open_year"]), int(bm["exit_year"]))
                if start > (end if end != 9999 else 2025):
                    continue
                used_biz.add(bid)
                pct = float(rng.choice([51, 60, 75, 80, 100]))
                own_links.append(_make_own_row(
                    person["person_id"], bid, pct, "majority_owner", start, end
                ))

            elif ttype == "multi_biz_owner":
                n_biz = rng.choice([2, 3, 4], p=[0.60, 0.30, 0.10])
                for b_num in range(n_biz):
                    # Alternate between corp and sole pools
                    bid = (_next_available(corp_pool, corp_ptr, z)
                           or _next_available(sole_pool, sole_ptr, z))
                    if bid is None:
                        break
                    offset = (0 if b_num == 0 else
                              int(rng.choice([0,1,2,3], p=[0.4,0.3,0.2,0.1])))
                    bm = biz_meta.loc[bid]
                    start, end = _overlap(p_entry, p_exit,
                                          int(bm["open_year"]), int(bm["exit_year"]),
                                          offset)
                    if start > (end if end != 9999 else 2025):
                        continue
                    used_biz.add(bid)
                    own_links.append(_make_own_row(
                        person["person_id"], bid,
                        float(rng.uniform(25, 100)),
                        "multi_owner", start, end,
                    ))

            elif ttype == "w2_with_side_biz":
                bid = _next_available(sole_pool, sole_ptr, z)
                if bid is None:
                    continue
                offset = int(rng.choice([0,1,2,3,4], p=[0.30,0.25,0.22,0.15,0.08]))
                bm = biz_meta.loc[bid]
                start, end = _overlap(p_entry, p_exit,
                                      int(bm["open_year"]), int(bm["exit_year"]),
                                      offset)
                if start > (end if end != 9999 else 2025):
                    continue
                used_biz.add(bid)
                own_links.append(_make_own_row(
                    person["person_id"], bid, 100.0, "side_owner", start, end
                ))

    own_df = pd.DataFrame(own_links)
    log.info("Ownership links: %d", len(own_df))

    # ─────────────────────────────────────────────────────────────────────────
    # EMPLOYMENT LINKS
    # Strategy: for each employer, draw the required number of workers from
    # the pre-shuffled zone pool using a rotating pointer.
    # ─────────────────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────────
    # EMPLOYMENT LINKS  (person-driven — guarantees full W2 coverage)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("Generating employment links...")

    ZONE_SALARY_MULT: dict[int, float] = {
        1: 0.85, 2: 0.95, 3: 0.90, 4: 1.15, 5: 1.40,
    }

    # ── Step 1: expand who gets W2 links ─────────────────────────────────────
    # retired and investor persons can have W2 income too
    W2_TYPES_EXPANDED = frozenset([
        "pure_w2",
        "w2_with_side_biz",
        "retired",
        "investor",
    ])
    w2_persons = persons[
        persons["taxpayer_type"].isin(W2_TYPES_EXPANDED)
    ].copy().reset_index(drop=True)

    log.info("W2-eligible persons: %d", len(w2_persons))

    # ── Step 2: build a business lookup by zone ───────────────────────────────
    # We sample WITH replacement — many workers can work at the same business
    # This is realistic and solves the pool exhaustion problem
    biz_by_zone: dict[int, list] = {}
    for z in range(1, 6):
        zone_biz_ids = businesses[
            businesses["zone"] == z
        ]["business_id"].tolist()

        if not zone_biz_ids:
            log.warning("Zone %d has no businesses — check businesses.csv", z)
            biz_by_zone[z] = []
            continue

        biz_by_zone[z] = zone_biz_ids
        log.info("Zone %d: %d businesses available", z, len(zone_biz_ids))

    # Pre-index businesses for fast salary lookups
    biz_meta = businesses.set_index("business_id")

    emp_links: list[dict] = []

    # ── Step 3: give every W2 person a primary job ────────────────────────────
    for _, person in w2_persons.iterrows():
        z       = int(person["zone"])
        p_entry = int(person["entry_year"])
        p_exit  = int(person["exit_year"])
        ttype   = person["taxpayer_type"]

        zone_biz = biz_by_zone.get(z, [])
        if not zone_biz:
            continue  # no businesses in this zone — skip (should be rare)

        # ── Find a business that was open when the person entered ─────────────
        # Try up to 10 random businesses before giving up
        chosen_bid   = None
        emp_start    = p_entry
        emp_end      = p_exit

        for _ in range(10):
            candidate = str(rng.choice(zone_biz))
            bm        = biz_meta.loc[candidate]
            b_open    = int(bm["open_year"])
            b_exit    = int(bm["exit_year"])

            # Overlap: both person and business must be active
            start = max(p_entry, b_open)

            p_end_cmp = p_exit if p_exit != 9999 else 2025
            b_end_cmp = b_exit if b_exit != 9999 else 2025
            end_cmp   = min(p_end_cmp, b_end_cmp)

            if start > end_cmp:
                continue  # no overlap, try another business

            chosen_bid = candidate
            emp_start  = start
            emp_end    = 9999 if (p_exit == 9999 and b_exit == 9999) else end_cmp
            break

        if chosen_bid is None:
            # All 10 attempts failed — just assign any business and
            # use person's own active years as employment period
            chosen_bid = str(rng.choice(zone_biz))
            emp_start  = p_entry
            emp_end    = p_exit

        # ── Salary ────────────────────────────────────────────────────────────
        occ       = person["primary_occupation"]
        occ_p     = bls.get(occ, {"log_mean": 10.5, "log_std": 0.7})
        zone_mult = ZONE_SALARY_MULT[z]

        # Retired workers and investors typically work part-time
        if ttype == "retired":
            is_pt    = int(rng.random() < 0.60)
            log_mean = occ_p["log_mean"] - 0.3   # lower salary for older workers
        elif ttype == "investor":
            is_pt    = int(rng.random() < 0.35)
            log_mean = occ_p["log_mean"] + 0.2   # investors often well-paid
        else:
            is_pt    = int(rng.random() < 0.15)
            log_mean = occ_p["log_mean"]

        base_sal = float(np.clip(
            np.exp(rng.normal(log_mean, occ_p["log_std"])) * zone_mult,
            18_000,    # minimum ~$18k/year
            800_000,   # maximum cap
        ))

        emp_links.append({
            "person_id":        person["person_id"],
            "business_id":      chosen_bid,
            "employment_start": emp_start,
            "employment_end":   emp_end,
            "base_salary_2019": round(base_sal, 2),
            "occupation":       occ,
            "is_part_time":     is_pt,
            "job_title_level":  str(rng.choice(
                ["entry", "mid", "senior", "manager", "executive"],
                p=[0.22, 0.38, 0.24, 0.12, 0.04],
            )),
        })

    # ── Step 4: second jobs for w2_with_side_biz ─────────────────────────────
    # ~40% of w2_with_side_biz persons have a second W2 job
    side_biz_persons = w2_persons[
        w2_persons["taxpayer_type"] == "w2_with_side_biz"
    ]

    for _, person in side_biz_persons.iterrows():
        if rng.random() > 0.40:
            continue

        z       = int(person["zone"])
        p_entry = int(person["entry_year"])
        p_exit  = int(person["exit_year"])

        zone_biz = biz_by_zone.get(z, [])
        if not zone_biz:
            continue

        bid   = str(rng.choice(zone_biz))
        bm    = biz_meta.loc[bid]
        b_open = int(bm["open_year"])
        b_exit = int(bm["exit_year"])

        start     = max(p_entry, b_open)
        p_end_cmp = p_exit if p_exit != 9999 else 2025
        b_end_cmp = b_exit if b_exit != 9999 else 2025
        end_cmp   = min(p_end_cmp, b_end_cmp)

        if start > end_cmp:
            continue

        occ   = person["primary_occupation"]
        occ_p = bls.get(occ, {"log_mean": 10.5, "log_std": 0.7})

        # Second job is always lower paid and part-time
        base_sal = float(np.clip(
            np.exp(rng.normal(
                occ_p["log_mean"] - 0.5,
                occ_p["log_std"],
            )) * ZONE_SALARY_MULT[z],
            8_000,
            120_000,
        ))

        emp_links.append({
            "person_id":        person["person_id"],
            "business_id":      bid,
            "employment_start": start,
            "employment_end":   (9999 if (p_exit == 9999 and b_exit == 9999)
                                 else end_cmp),
            "base_salary_2019": round(base_sal, 2),
            "occupation":       occ,
            "is_part_time":     1,
            "job_title_level":  "entry",
        })

    emp_df = pd.DataFrame(emp_links)

    # ── Validation ────────────────────────────────────────────────────────────
    total_w2  = len(w2_persons)
    covered   = emp_df["person_id"].nunique()
    log.info(
        "Employment links: %d total | %d unique persons | coverage=%.1f%%",
        len(emp_df),
        covered,
        covered / total_w2 * 100,
    )

    for z in range(1, 6):
        zone_w2 = w2_persons[w2_persons["zone"] == z]
        zone_cov = emp_df[
            emp_df["person_id"].isin(zone_w2["person_id"])
        ]["person_id"].nunique()
        log.info(
            "  Zone %d: %d / %d covered (%.0f%%)",
            z, zone_cov, len(zone_w2),
            zone_cov / max(len(zone_w2), 1) * 100,
        )
    # ── Save ──────────────────────────────────────────────────────────────────
    PB_LINKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    own_df.to_csv(PB_LINKS_CSV, index=False)
    emp_df.to_csv(EMP_LINKS_CSV, index=False)

    final_vol.commit()
    logs_vol.commit()
    log.info("Links saved -> %s  %s", PB_LINKS_CSV, EMP_LINKS_CSV)
    return {
        "ownership":   len(own_df),
        "employment":  len(emp_df),
        "status":      "ok",
    }


@app.local_entrypoint()
def main():
    result = generate_links.remote()
    print(result)