# 03_generate_businesses.py
"""
Generate businesses.csv — 110,000 synthetic businesses.
Fully vectorized. Imports from config/utils only.
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

app      = modal.App("taxxx-pipeline-03-businesses")
dist_vol  = modal.Volume.from_name(VOLUME_NAMES["dists"],  create_if_missing=True)
final_vol = modal.Volume.from_name(VOLUME_NAMES["final"],  create_if_missing=True)
logs_vol  = modal.Volume.from_name(VOLUME_NAMES["logs"],   create_if_missing=True)



VOLUMES = {
    "/distributions": dist_vol,   # DIST_BASE  — master_distributions.pkl lives here
    "/final_dataset": final_vol,  # FINAL_BASE — persons.csv written here
    "/logs":          logs_vol,   # LOGS_BASE
}
# ── Module-level constants ────────────────────────────────────────────────────
ENTITY_TYPES = {
    "sole_prop":   0.412,
    "llc_single":  0.198,
    "llc_multi":   0.089,
    "scorp":       0.148,
    "partnership": 0.062,
    "ccorp":       0.058,
    "nonprofit":   0.033,
}  # sum = 1.000 ✓

# Year-by-year failure rates by industry (calibrated to SBA / BLS data)
# 2020 spike reflects COVID-19 closures
FAILURE_RATES: dict[str, dict[int, float]] = {
    "food_service":          {2019:0.17,2020:0.28,2021:0.14,2022:0.12,2023:0.11,2024:0.10,2025:0.10},
    "retail_trade":          {2019:0.13,2020:0.22,2021:0.11,2022:0.09,2023:0.09,2024:0.08,2025:0.09},
    "construction":          {2019:0.10,2020:0.14,2021:0.08,2022:0.07,2023:0.07,2024:0.07,2025:0.08},
    "professional_services": {2019:0.07,2020:0.11,2021:0.06,2022:0.05,2023:0.05,2024:0.05,2025:0.05},
    "healthcare_solo":       {2019:0.05,2020:0.08,2021:0.04,2022:0.04,2023:0.04,2024:0.03,2025:0.04},
    "real_estate_rental":    {2019:0.06,2020:0.09,2021:0.05,2022:0.06,2023:0.05,2024:0.05,2025:0.05},
    "transportation_gig":    {2019:0.11,2020:0.09,2021:0.08,2022:0.09,2023:0.08,2024:0.08,2025:0.08},
    "information_tech":      {2019:0.08,2020:0.10,2021:0.06,2022:0.05,2023:0.05,2024:0.05,2025:0.05},
}
_DEFAULT_FAILURE_RATE = 0.09

BIZ_FRAUD_TYPES = {
    "clean":                          0.830,
    "cash_skimming":                  0.048,
    "revenue_suppression":            0.031,
    "inflated_cogs":                  0.028,
    "fictitious_deductions":          0.022,
    "payroll_underreporting":         0.012,
    "worker_misclassification":       0.009,
    "shell_company_income_shifting":  0.006,
    "low_salary_scorp":               0.014,
}  # sum = 1.000 ✓

# Maps ZONE_PROFILES industry_mix keys → FAILURE_RATES / schedule_c keys
INDUSTRY_MAP: dict[str, str] = {
    "retail":                "retail_trade",
    "retail_trade":          "retail_trade",
    "food_service":          "food_service",
    "construction":          "construction",
    "professional_services": "professional_services",
    "healthcare":            "healthcare_solo",
    "healthcare_solo":       "healthcare_solo",
    "real_estate":           "real_estate_rental",
    "real_estate_rental":    "real_estate_rental",
    "tech":                  "information_tech",
    "information_tech":      "information_tech",
    "finance":               "professional_services",
    "finance_insurance":     "professional_services",
    "manufacturing":         "retail_trade",
    "transportation":        "transportation_gig",
    "transportation_gig":    "transportation_gig",
    "agriculture":           "retail_trade",
    "hospitality_tourism":   "food_service",
    "other":                 "professional_services",
}

# Log-normal parameters for employee count by industry
EMP_PARAMS: dict[str, dict[str, float]] = {
    "food_service":          {"lm": 1.8, "ls": 0.9},
    "retail_trade":          {"lm": 1.6, "ls": 0.8},
    "construction":          {"lm": 1.5, "ls": 0.9},
    "professional_services": {"lm": 0.9, "ls": 0.8},
    "healthcare_solo":       {"lm": 1.2, "ls": 0.7},
    "real_estate_rental":    {"lm": 0.4, "ls": 0.7},
    "transportation_gig":    {"lm": 0.3, "ls": 0.6},
    "information_tech":      {"lm": 1.1, "ls": 0.8},
}
_DEFAULT_EMP_PARAMS = {"lm": 1.0, "ls": 0.8}

# Industries where cash transactions dominate
_CASH_HEAVY_INDUSTRIES = frozenset({
    "food_service", "retail_trade", "construction",
    "transportation_gig", "real_estate_rental",
})


def _make_business_id(global_idx: int, zone: int, entity_type: str) -> str:
    """B<zone>_<entity_abbrev>_<zero-padded-7-digit-index>"""
    abbrev = {
        "sole_prop": "SP", "llc_single": "LS", "llc_multi": "LM",
        "scorp": "SC", "partnership": "PT", "ccorp": "CC", "nonprofit": "NP",
    }.get(entity_type, "XX")
    return f"B{zone}_{abbrev}_{global_idx:07d}"


@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=4,
    memory=8192,
    timeout=1800,
)
def generate_businesses():
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"   # triggers IS_MODAL=True in config

    from config import (
        ZONE_PROFILES, RANDOM_SEED,
        BUSINESSES_CSV, N_BIZ_TOTAL,
    )
    from utils import get_logger, load_distributions

    log = get_logger("03_businesses", "03_businesses.log")
    rng = np.random.default_rng(RANDOM_SEED + 1)

    log.info("Loading distributions...")
    dist  = load_distributions()
    # schedule_c dist keyed by mapped industry name; fallback to empty dict
    sch_c = dist.get("schedule_c", {})

    BUSINESSES_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ── Cohort sizes ──────────────────────────────────────────────────────────
    COHORTS = {
        2019: 82_000,
        2020:  4_800,
        2021:  7_200,
        2022:  5_100,
        2023:  4_900,
        2024:  3_800,
        2025:  2_200,
    }
    assert sum(COHORTS.values()) == N_BIZ_TOTAL, (
        f"Cohort sum {sum(COHORTS.values())} != N_BIZ_TOTAL {N_BIZ_TOTAL}"
    )

    ZONE_KEYS   = list(ZONE_PROFILES.keys())                      # [1,2,3,4,5]
    BIZ_ZONE_W  = np.array([0.30, 0.25, 0.22, 0.14, 0.09])      # sum=1.00 ✓

    ET_LABELS   = list(ENTITY_TYPES.keys())
    ET_BASE_W   = np.array(list(ENTITY_TYPES.values()), dtype=float)

    BFT_LABELS  = list(BIZ_FRAUD_TYPES.keys())
    BFT_BASE_W  = np.array(list(BIZ_FRAUD_TYPES.values()), dtype=float)

    # Compute avg_fraud from config rather than hard-coding
    avg_fraud = sum(
        ZONE_PROFILES[z]["fraud_base_rate"] * ZONE_PROFILES[z]["population_share"]
        for z in ZONE_KEYS
    )

    # ── Vectorized helpers ────────────────────────────────────────────────────

    def _vectorized_choice(
        rng: np.random.Generator,
        labels: list,
        weight_matrix: np.ndarray,
    ) -> np.ndarray:
        """One label per row; weight_matrix shape (n, k)."""
        wm   = weight_matrix / weight_matrix.sum(axis=1, keepdims=True)
        cumw = wm.cumsum(axis=1)
        cumw[:, -1] = 1.0
        rand = rng.random(len(wm))
        idx  = np.array(
            [np.searchsorted(cumw[i], rand[i]) for i in range(len(rand))]
        )
        idx  = np.clip(idx, 0, len(labels) - 1)
        return np.array(labels)[idx]

    def _assign_industries_vectorized(
        zones: np.ndarray,
    ) -> np.ndarray:
        """
        Assign a mapped industry string to each business based on zone
        industry_mix. Vectorized by zone group.
        """
        result = np.empty(len(zones), dtype=object)
        for z in ZONE_KEYS:
            mask = zones == z
            if not mask.any():
                continue
            mix  = ZONE_PROFILES[z]["industry_mix"]
            keys = list(mix.keys())
            wts  = np.array([mix[k] for k in keys], dtype=float)
            wts /= wts.sum()
            n_z  = mask.sum()
            raw  = rng.choice(keys, size=n_z, p=wts)
            result[mask] = np.array(
                [INDUSTRY_MAP.get(r, "professional_services") for r in raw]
            )
        return result

    def _assign_entity_types_vectorized(
        industries: np.ndarray,
    ) -> np.ndarray:
        """
        Assign entity type per business with industry-specific adjustments.
        Vectorized by unique industry group.
        """
        result = np.empty(len(industries), dtype=object)
        for ind in np.unique(industries):
            mask = industries == ind
            w    = ET_BASE_W.copy()
            if ind == "healthcare_solo":
                w[ET_LABELS.index("scorp")]      *= 2.2
            elif ind == "construction":
                w[ET_LABELS.index("llc_single")] *= 1.6
            elif ind == "food_service":
                w[ET_LABELS.index("sole_prop")]  *= 1.4
            elif ind == "information_tech":
                w[ET_LABELS.index("scorp")]      *= 1.8
                w[ET_LABELS.index("llc_multi")]  *= 1.4
            w /= w.sum()
            n_ind = mask.sum()
            result[mask] = rng.choice(ET_LABELS, size=n_ind, p=w)
        return result

    def _assign_employee_counts_vectorized(
        industries: np.ndarray,
        entity_types: np.ndarray,
    ) -> np.ndarray:
        """Log-normal employee count, capped by entity type."""
        n      = len(industries)
        lm_arr = np.array(
            [EMP_PARAMS.get(ind, _DEFAULT_EMP_PARAMS)["lm"] for ind in industries]
        )
        ls_arr = np.array(
            [EMP_PARAMS.get(ind, _DEFAULT_EMP_PARAMS)["ls"] for ind in industries]
        )
        raw = np.exp(rng.normal(lm_arr, ls_arr)).astype(int)

        # Cap by entity type
        cap = np.where(
            entity_types == "sole_prop", 4,
            np.where(entity_types == "scorp", 25, 200)
        )
        return np.clip(raw, 0, cap)

    def _assign_fraud_types_vectorized(
        industries: np.ndarray,
        entity_types: np.ndarray,
        zones: np.ndarray,
        sch_c: dict,
    ) -> np.ndarray:
        """
        Assign business fraud type with zone + industry + entity adjustments.
        Vectorized by unique (industry, entity_type, zone) combination.
        """
        n      = len(industries)
        w_mat  = np.tile(BFT_BASE_W, (n, 1)).copy()

        # Zone scaling (vectorized)
        for z in ZONE_KEYS:
            mask  = zones == z
            if not mask.any():
                continue
            scale = ZONE_PROFILES[z]["fraud_base_rate"] / avg_fraud
            fraud_cols = [
                j for j, lbl in enumerate(BFT_LABELS) if lbl != "clean"
            ]
            w_mat[np.ix_(mask, fraud_cols)] *= scale

        # Cash intensity from schedule_c dist
        i_skim = BFT_LABELS.index("cash_skimming")
        i_rev  = BFT_LABELS.index("revenue_suppression")
        i_pay  = BFT_LABELS.index("payroll_underreporting")
        i_mis  = BFT_LABELS.index("worker_misclassification")
        i_sc   = BFT_LABELS.index("low_salary_scorp")

        for ind in np.unique(industries):
            mask      = industries == ind
            cash_int  = sch_c.get(ind, {}).get("cash_intensity", 0.2)
            if cash_int > 0.35:
                w_mat[mask, i_skim] *= 2.4
                w_mat[mask, i_rev]  *= 1.8

        # S-corp salary suppression
        scorp_mask = entity_types == "scorp"
        w_mat[scorp_mask, i_sc] *= 4.5

        # High cash-density zones → payroll / misclassification risk
        for z in ZONE_KEYS:
            if ZONE_PROFILES[z]["cash_business_density"] > 0.35:
                mask = zones == z
                w_mat[mask, i_pay] *= 1.8
                w_mat[mask, i_mis] *= 1.6

        w_mat = np.clip(w_mat, 1e-9, None)
        return _vectorized_choice(rng, BFT_LABELS, w_mat)

    def _assign_exit_years_vectorized(
        open_year: int,
        industries: np.ndarray,
        entity_types: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """
        Simulate year-by-year failure for each business.
        Returns exit year (open_year <= exit <= 2025) or 9999 if still active.
        Vectorized across businesses for each calendar year.
        """
        # Survival multiplier: corps are more resilient, sole props less so
        adj = np.where(
            np.isin(entity_types, ["ccorp", "scorp"]), 0.6,
            np.where(entity_types == "sole_prop", 1.3, 1.0)
        )
        exit_years = np.full(n, 9999, dtype=int)
        alive      = np.ones(n, dtype=bool)

        for yr in range(open_year, 2026):
            if not alive.any():
                break
            # Failure probability for each still-alive business this year
            fr_arr = np.array(
                [
                    FAILURE_RATES.get(ind, {}).get(yr, _DEFAULT_FAILURE_RATE)
                    for ind in industries
                ]
            ) * adj
            draws  = rng.random(n)
            failed = alive & (draws < fr_arr)
            exit_years[failed] = yr
            alive[failed]      = False

        return exit_years

    # ── Main cohort loop ──────────────────────────────────────────────────────
    all_cohorts: list[pd.DataFrame] = []
    global_idx = 0

    for open_year, n in COHORTS.items():
        log.info("Cohort %d: %d businesses", open_year, n)

        zones        = rng.choice(ZONE_KEYS, size=n, p=BIZ_ZONE_W)
        industries   = _assign_industries_vectorized(zones)
        entity_types = _assign_entity_types_vectorized(industries)
        n_employees  = _assign_employee_counts_vectorized(industries, entity_types)
        biz_fraud    = _assign_fraud_types_vectorized(
            industries, entity_types, zones, sch_c
        )
        exit_years   = _assign_exit_years_vectorized(
            open_year, industries, entity_types, n
        )

        # Cash acceptance: cash-heavy industries always + some others
        accepts_cash = np.where(
            np.isin(industries, list(_CASH_HEAVY_INDUSTRIES)),
            1,
            (rng.random(n) < 0.45).astype(int),
        )
        has_pos = (rng.random(n) > 0.28).astype(int)

        pids = [
            _make_business_id(global_idx + i, int(zones[i]), entity_types[i])
            for i in range(n)
        ]

        cohort_df = pd.DataFrame({
            "business_id":         pids,
            "zone":                zones.astype(int),
            "open_year":           open_year,
            "exit_year":           exit_years,
            "entity_type":         entity_types,
            "industry":            industries,
            "n_employees_base":    n_employees,
            "business_fraud_type": biz_fraud,
            "has_pos_system":      has_pos,
            "accepts_cash":        accepts_cash,
        })
        all_cohorts.append(cohort_df)
        global_idx += n

    biz = pd.concat(all_cohorts, ignore_index=True)

    assert len(biz) == N_BIZ_TOTAL, (
        f"Expected {N_BIZ_TOTAL} businesses, got {len(biz)}"
    )

    log.info("Total businesses: %d", len(biz))
    log.info(
        "Entity types:\n%s",
        biz["entity_type"].value_counts(normalize=True).round(3).to_string(),
    )
    log.info(
        "Fraud types:\n%s",
        biz["business_fraud_type"].value_counts(normalize=True).round(4).to_string(),
    )

    biz.to_csv(BUSINESSES_CSV, index=False)
    final_vol.commit()
    logs_vol.commit()
    log.info("Saved %d businesses -> %s", len(biz), BUSINESSES_CSV)
    return {"rows": len(biz), "status": "ok"}


@app.local_entrypoint()
def main():
    result = generate_businesses.remote()
    print(result)