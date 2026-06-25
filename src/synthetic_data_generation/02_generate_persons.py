# 02_generate_persons.py
"""
Generate master persons.csv — 460,000 synthetic individuals.
Vectorized — no per-person Python loops.
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

app       = modal.App("taxxx-pipeline-02-persons")
dist_vol  = modal.Volume.from_name(VOLUME_NAMES["dists"], create_if_missing=True)
final_vol = modal.Volume.from_name(VOLUME_NAMES["final"], create_if_missing=True)
logs_vol  = modal.Volume.from_name(VOLUME_NAMES["logs"],  create_if_missing=True)

VOLUMES = {
    "/distributions": dist_vol,
    "/final_dataset": final_vol,
    "/logs":          logs_vol,
}

# ── Fraud personas ────────────────────────────────────────────────────────────
FRAUD_PERSONAS: dict[str, float] = {
    "clean":          0.68,
    "chronic_evader": 0.065,
    "opportunistic":  0.085,
    "late_onset":     0.045,
    "escalating":     0.055,
    "one_time":       0.070,
}

# ── Fraud compatibility map (names match panels.py elif branches exactly) ─────
FRAUD_COMPATIBILITY: dict[str, list[str]] = {
    "pure_w2": [
        "fictitious_deductions",
        "1099_not_reported",
    ],
    "pure_se": [
        "unreported_cash_income",
        "cash_skimming",
        "expense_recharacterization",
        "inflated_cogs",
        "fictitious_deductions",
    ],
    "w2_with_side_biz": [
        "1099_not_reported",
        "expense_recharacterization",
        "fictitious_deductions",
    ],
    "business_owner": [
        "unreported_cash_income",
        "expense_recharacterization",
        "cash_skimming",
        "revenue_suppression",
        "payroll_underreporting",
        "fictitious_deductions",
    ],
    "multi_biz_owner": [
        "unreported_cash_income",
        "expense_recharacterization",
        "cash_skimming",
        "payroll_underreporting",
        "worker_misclassification",
        "shell_company_income_shifting",
        "fictitious_deductions",
    ],
    "gig_only": [
        "gig_income_omitted",
        "expense_recharacterization",
        "fictitious_deductions",
    ],
    "retired": [
        "fictitious_deductions",
        "1099_not_reported",
    ],
    "investor": [
        "capital_gains_omit",
        "fictitious_deductions",
        "offshore_hidden",
    ],
}


def _make_person_id(global_idx: int, zone: int) -> str:
    return f"P{zone}_{global_idx:07d}"


@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=4,
    memory=16384,
    timeout=1800,
)
def generate_persons():
    import sys
    import os
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    from config import (
        ZONE_PROFILES, TAXPAYER_TYPES,
        RANDOM_SEED, PERSONS_CSV,
        N_PERSONS_TOTAL,
    )
    from utils import get_logger, load_distributions

    log = get_logger("02_persons", "02_persons.log")
    rng = np.random.default_rng(RANDOM_SEED)

    log.info("Loading distributions...")
    dist = load_distributions()

    PERSONS_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ── Cohort sizes ──────────────────────────────────────────────────────────
    COHORTS = {
        2019: 360_000,
        2020:  16_500,
        2021:  17_200,
        2022:  15_800,
        2023:  16_900,
        2024:  17_400,
        2025:  16_200,
    }
    assert sum(COHORTS.values()) == N_PERSONS_TOTAL, (
        f"Cohort sum {sum(COHORTS.values())} != N_PERSONS_TOTAL {N_PERSONS_TOTAL}"
    )

    # ── Static label arrays ───────────────────────────────────────────────────
    ZONE_KEYS    = list(ZONE_PROFILES.keys())
    ZONE_WEIGHTS = np.array(
        [ZONE_PROFILES[z]["population_share"] for z in ZONE_KEYS]
    )

    EDU_LABELS = [
        "no_hs", "hs_diploma", "some_college", "associates",
        "bachelors", "masters", "professional", "doctorate",
    ]
    EDU_BASE_W = np.array([0.09, 0.27, 0.20, 0.09, 0.22, 0.09, 0.02, 0.02])

    FS_LABELS = [
        "single", "married_joint", "married_separate",
        "head_of_household", "qualifying_widow",
    ]

    OCC_LABELS  = list(dist["bls"]["occupations"].keys())
    OCC_WEIGHTS = np.array(
        [dist["bls"]["occupations"][o]["emp_share"] for o in OCC_LABELS],
        dtype=float,
    )
    OCC_WEIGHTS /= OCC_WEIGHTS.sum()

    T_LABELS  = list(TAXPAYER_TYPES.keys())
    T_WEIGHTS = np.array(list(TAXPAYER_TYPES.values()), dtype=float)

    FP_LABELS  = list(FRAUD_PERSONAS.keys())
    FP_WEIGHTS = np.array(list(FRAUD_PERSONAS.values()), dtype=float)

    # ── Vectorized helpers ────────────────────────────────────────────────────

    def _vectorized_choice(
        rng: np.random.Generator,
        labels: list,
        weight_matrix: np.ndarray,
    ) -> np.ndarray:
        weight_matrix = weight_matrix / weight_matrix.sum(axis=1, keepdims=True)
        cumw = weight_matrix.cumsum(axis=1)
        cumw[:, -1] = 1.0
        rand = rng.random(len(weight_matrix))
        idx  = np.array(
            [np.searchsorted(cumw[i], rand[i]) for i in range(len(rand))]
        )
        idx = np.clip(idx, 0, len(labels) - 1)
        return np.array(labels)[idx]

    def _edu_weights(ages: np.ndarray) -> np.ndarray:
        w     = np.tile(EDU_BASE_W, (len(ages), 1)).copy()
        young = ages < 24
        old   = ages > 55
        w[young] = [0.08, 0.22, 0.35, 0.12, 0.18, 0.04, 0.005, 0.005]
        w[old]   = [0.12, 0.32, 0.18, 0.08, 0.18, 0.08, 0.015, 0.015]
        return w

    def _filing_status_weights(ages: np.ndarray) -> np.ndarray:
        n = len(ages)
        w = np.zeros((n, 5), dtype=float)
        #                              single  mj     ms    hoh   qw
        w[ages < 26]                = [0.72,  0.14,  0.01, 0.12, 0.01]
        w[(ages >= 26) & (ages < 40)]=[0.38,  0.44,  0.03, 0.14, 0.01]
        w[(ages >= 40) & (ages < 65)]=[0.32,  0.47,  0.03, 0.15, 0.03]
        w[(ages >= 65) & (ages < 75)]=[0.38,  0.44,  0.02, 0.10, 0.06]
        w[ages >= 75]               = [0.42,  0.38,  0.02, 0.10, 0.08]
        return w

    def _taxpayer_weights(
        ages:  np.ndarray,
        edus:  np.ndarray,
        zones: np.ndarray,
    ) -> np.ndarray:
        n = len(ages)
        w = np.tile(T_WEIGHTS, (n, 1)).copy()

        i_gig = T_LABELS.index("gig_only")
        i_ret = T_LABELS.index("retired")
        i_mbo = T_LABELS.index("multi_biz_owner")
        i_wse = T_LABELS.index("w2_with_side_biz")
        i_bo  = T_LABELS.index("business_owner")
        i_inv = T_LABELS.index("investor")
        i_w2  = T_LABELS.index("pure_w2")
        i_se  = T_LABELS.index("pure_se")

        young = ages < 28
        prime = (ages >= 35) & (ages <= 55)
        old   = ages > 60

        w[young, i_gig] *= 2.2;  w[young, i_ret] *= 0.05
        w[young, i_mbo] *= 0.3
        w[old,   i_ret] *= 3.5;  w[old,   i_inv] *= 2.1
        w[old,   i_gig] *= 0.4;  w[old,   i_w2]  *= 0.7
        w[prime, i_bo]  *= 1.8;  w[prime, i_mbo] *= 2.1
        w[prime, i_wse] *= 1.4

        high_edu = np.isin(edus, ["bachelors", "masters", "professional", "doctorate"])
        low_edu  = np.isin(edus, ["no_hs", "hs_diploma"])
        w[high_edu, i_bo]  *= 1.4;  w[high_edu, i_inv] *= 1.6
        w[high_edu, i_w2]  *= 1.2
        w[low_edu,  i_gig] *= 1.5;  w[low_edu,  i_se]  *= 1.3
        w[low_edu,  i_inv] *= 0.4

        rural = zones == 3
        urban = zones == 1
        tech  = zones == 4
        w[rural, i_se]  *= 1.5;  w[rural, i_gig] *= 0.7
        w[urban, i_inv] *= 1.8;  w[urban, i_mbo] *= 1.6
        w[tech,  i_bo]  *= 1.6;  w[tech,  i_mbo] *= 1.8

        w = np.clip(w, 1e-9, None)
        return w

    def _fraud_persona_weights(
        ttypes: np.ndarray,
        zones:  np.ndarray,
        ages:   np.ndarray,
    ) -> np.ndarray:
        n = len(ttypes)
        w = np.tile(FP_WEIGHTS, (n, 1)).copy()

        i_ch  = FP_LABELS.index("chronic_evader")
        i_op  = FP_LABELS.index("opportunistic")
        i_lo  = FP_LABELS.index("late_onset")
        i_esc = FP_LABELS.index("escalating")
        i_ot  = FP_LABELS.index("one_time")

        avg_fraud = sum(
            ZONE_PROFILES[z]["fraud_base_rate"] * ZONE_PROFILES[z]["population_share"]
            for z in ZONE_KEYS
        )
        for z in ZONE_KEYS:
            mask = zones == z
            if not mask.any():
                continue
            scale      = ZONE_PROFILES[z]["fraud_base_rate"] / avg_fraud
            fraud_cols = [j for j, lbl in enumerate(FP_LABELS) if lbl != "clean"]
            w[np.ix_(mask, fraud_cols)] *= scale

        biz_mask = np.isin(ttypes, ["business_owner", "multi_biz_owner", "pure_se"])
        w[biz_mask, i_ch]  *= 1.8
        w[biz_mask, i_op]  *= 1.5
        w[biz_mask, i_esc] *= 1.6

        old_mask = ages > 55
        w[old_mask, i_lo]  *= 0.4
        w[old_mask, i_esc] *= 0.5

        young_mask = ages < 30
        w[young_mask, i_ot] *= 1.8
        w[young_mask, i_ch] *= 0.6

        w = np.clip(w, 1e-9, None)
        return w

    def _primary_fraud_vectorized(
        ttypes:  np.ndarray,
        fpersns: np.ndarray,
        rng:     np.random.Generator,
    ) -> np.ndarray:
        result     = np.full(len(ttypes), "none", dtype=object)
        fraud_mask = fpersns != "clean"
        for ttype in np.unique(ttypes[fraud_mask]):
            compatible = FRAUD_COMPATIBILITY.get(ttype, [])
            if not compatible:
                continue
            sub_mask        = fraud_mask & (ttypes == ttype)
            result[sub_mask] = rng.choice(compatible, size=sub_mask.sum())
        return result

    def _exit_years_vectorized(
        ages:       np.ndarray,
        entry_year: int,
        rng:        np.random.Generator,
    ) -> np.ndarray:
        scale    = np.where(ages > 70, 4.5, np.where(ages > 55, 10.2, 22.0))
        survival = rng.exponential(scale=scale).astype(int)
        ey       = entry_year + survival
        return np.where(ey <= 2025, ey, 9999)

    # ── Main generation loop ──────────────────────────────────────────────────
    all_cohorts: list[pd.DataFrame] = []
    global_idx = 0

    for entry_year, n in COHORTS.items():
        log.info("Cohort %d: %d persons", entry_year, n)

        # ── Sample all arrays for this cohort ─────────────────────────────────
        ages  = np.clip(
            rng.normal(42.1 if entry_year == 2019 else 28.4,
                       13.8 if entry_year == 2019 else 7.2,
                       n),
            18,
            80 if entry_year == 2019 else 75,
        ).astype(int)

        zones       = rng.choice(ZONE_KEYS, size=n, p=ZONE_WEIGHTS)
        sex         = rng.choice(["M", "F"], size=n, p=[0.488, 0.512])
        edus        = _vectorized_choice(rng, EDU_LABELS, _edu_weights(ages))
        fstats      = _vectorized_choice(rng, FS_LABELS,  _filing_status_weights(ages))
        ttypes      = _vectorized_choice(rng, T_LABELS,   _taxpayer_weights(ages, edus, zones))
        fpersns     = _vectorized_choice(rng, FP_LABELS,  _fraud_persona_weights(ttypes, zones, ages))
        primary_fraud = _primary_fraud_vectorized(ttypes, fpersns, rng)

        # ── one_time_target_year ──────────────────────────────────────────────
        # Computed HERE — before the DataFrame is built — so the variable
        # exists when it is referenced in the dict constructor below.
        # one_time personas get a single deterministic target year stored
        # at person-creation time.  _should_evade() in panels.py reads this
        # column instead of re-rolling the RNG on every call (which caused
        # the same person to "evade" in multiple years).
        # Everyone else gets -1, which never matches a real tax year.
        one_time_target_year = np.where(
            fpersns == "one_time",
            entry_year + rng.integers(0, 7, n),
            -1,
        )

        # ── Zone-aware boolean flags ──────────────────────────────────────────
        foreign_prob        = np.array([ZONE_PROFILES[z]["foreign_account_prevalence"] for z in zones])
        crypto_prob         = np.array([ZONE_PROFILES[z]["crypto_adoption"]            for z in zones])
        rental_prob         = np.array([ZONE_PROFILES[z]["rental_market_size"]         for z in zones])
        has_foreign_account = (rng.random(n) < foreign_prob).astype(int)
        is_crypto_user      = (rng.random(n) < crypto_prob).astype(int)
        has_rental_property = (rng.random(n) < rental_prob).astype(int)

        # ── Occupation ────────────────────────────────────────────────────────
        occupations = rng.choice(OCC_LABELS, size=n, p=OCC_WEIGHTS)

        # ── Risk score ────────────────────────────────────────────────────────
        risk = np.full(n, 15.0)
        risk[np.isin(ttypes, ["business_owner", "multi_biz_owner"])] += 18
        risk[ttypes == "pure_se"]                                     += 12
        risk[fpersns == "chronic_evader"]                             += 35
        risk[np.isin(fpersns, ["escalating", "opportunistic"])]       += 20
        risk[has_foreign_account == 1]                                += 22
        risk[is_crypto_user == 1]                                     += 8
        risk += rng.normal(0, 6, n)
        risk  = np.clip(risk, 1, 99).round(2)

        # ── Exit years ────────────────────────────────────────────────────────
        exit_years = _exit_years_vectorized(ages, entry_year, rng)

        # ── Person IDs ────────────────────────────────────────────────────────
        pids = [
            _make_person_id(global_idx + i, int(zones[i]))
            for i in range(n)
        ]

        # ── Build DataFrame ───────────────────────────────────────────────────
        # All arrays are defined above this point — no forward references.
        cohort_df = pd.DataFrame({
            "person_id":              pids,
            "zone":                   zones.astype(int),
            "entry_year":             entry_year,
            "exit_year":              exit_years,
            "age_at_entry":           ages,
            "sex":                    sex,
            "education":              edus,
            "filing_status":          fstats,
            "primary_occupation":     occupations,
            "taxpayer_type":          ttypes,
            "fraud_persona":          fpersns,
            "primary_fraud_type":     primary_fraud,
            "has_foreign_account":    has_foreign_account,
            "is_crypto_user":         is_crypto_user,
            "has_rental_property":    has_rental_property,
            "risk_score_base":        risk,
            "one_time_target_year":   one_time_target_year,  # ← in constructor
        })

        all_cohorts.append(cohort_df)   # append after df is fully built
        global_idx += n

    # ── Concatenate all cohorts ───────────────────────────────────────────────
    persons = pd.concat(all_cohorts, ignore_index=True)

    # ── Validation ────────────────────────────────────────────────────────────
    assert len(persons) == N_PERSONS_TOTAL, (
        f"Expected {N_PERSONS_TOTAL} persons, got {len(persons)}"
    )

    log.info("Total persons: %d", len(persons))
    log.info(
        "Fraud persona dist:\n%s",
        persons["fraud_persona"].value_counts(normalize=True).round(4).to_string(),
    )
    log.info(
        "Taxpayer type dist:\n%s",
        persons["taxpayer_type"].value_counts(normalize=True).round(3).to_string(),
    )
    log.info(
        "Zone dist:\n%s",
        persons["zone"].value_counts(normalize=True).round(3).to_string(),
    )
    log.info(
        "Filing status dist:\n%s",
        persons["filing_status"].value_counts(normalize=True).round(3).to_string(),
    )
    log.info(
        "one_time_target_year sample (first 10 one_time persons):\n%s",
        persons[persons["fraud_persona"] == "one_time"]["one_time_target_year"]
        .head(10).to_string(),
    )

    # ── Write output ──────────────────────────────────────────────────────────
    persons.to_csv(PERSONS_CSV, index=False)
    final_vol.commit()
    logs_vol.commit()
    log.info("Saved %d persons -> %s", len(persons), PERSONS_CSV)
    return {"rows": len(persons), "status": "ok"}


@app.local_entrypoint()
def main():
    result = generate_persons.remote()
    print(result)