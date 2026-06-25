# 05_generate_panels.py
"""
Generate all person-year panel records.
Writes 35 zone-year parquets + 7 by-year parquets.
Parallelized: one Modal container per zone-year.
"""

import modal
import numpy as np
import pandas as pd
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

app       = modal.App("taxxx-pipeline-05-panels")
dist_vol  = modal.Volume.from_name(VOLUME_NAMES["dists"],  create_if_missing=True)
final_vol = modal.Volume.from_name(VOLUME_NAMES["final"],  create_if_missing=True)
logs_vol  = modal.Volume.from_name(VOLUME_NAMES["logs"],   create_if_missing=True)

VOLUMES = {
    "/distributions": dist_vol,
    "/final_dataset": final_vol,
    "/logs":          logs_vol,
}

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

FRAUD_SCHEMA: dict[str, str] = {
    "unreported_cash_income":        "income_hiding",
    "1099_not_reported":             "income_hiding",
    "gig_income_omitted":            "income_hiding",
    "rental_income_hidden":          "income_hiding",
    "farm_income_hidden":            "income_hiding",
    "crypto_unreported":             "income_hiding",
    "offshore_hidden":               "income_hiding",
    "fictitious_deductions":         "deduction_fraud",
    "expense_recharacterization":    "deduction_fraud",
    "inflated_cogs":                 "deduction_fraud",
    "low_salary_scorp":              "payroll_evasion",
    "payroll_underreporting":        "payroll_evasion",
    "worker_misclassification":      "payroll_evasion",
    "revenue_suppression":           "business_suppression",
    "cash_skimming":                 "business_suppression",
    "shell_company_income_shifting": "business_suppression",
    "capital_gains_omit":            "income_hiding",
    "none":                          "none",
}



MKT_RETURNS: dict[int, float] = {
    2019:  0.288, 2020:  0.162, 2021:  0.269,
    2022: -0.196, 2023:  0.244, 2024:  0.235, 2025:  0.089,
}

FED_RATE_PROXY: dict[int, float] = {
    2019: 0.021, 2020: 0.007, 2021: 0.003,
    2022: 0.016, 2023: 0.051, 2024: 0.053, 2025: 0.045,
}

OCC_GROWTH: dict[str, float] = {
    "computer_math":    0.040, "management":      0.030,
    "legal":            0.025, "healthcare_pract": 0.028,
    "business_finance": 0.032, "food_service":    0.018,
    "building_grounds": 0.019, "personal_care":   0.020,
    "farming_fishing":  0.015,
}
_DEFAULT_OCC_GROWTH = 0.024

PORTFOLIO_MEDIANS: dict[str, float] = {
    "investor":        850_000,
    "business_owner":  320_000,
    "multi_biz_owner": 580_000,
    "retired":         420_000,
}
_DEFAULT_PORTFOLIO_MEDIAN = 55_000

EITC_MAX: dict[int, float] = {0: 538, 1: 3_584, 2: 5_920, 3: 6_660}


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _wage_growth(
    base: float,
    year: int,
    occupation: str,
    fraud_persona: str,
    rng: np.random.Generator,
    macro_y: dict,
) -> float:
    occ_adj     = OCC_GROWTH.get(occupation, _DEFAULT_OCC_GROWTH)
    base_growth = (
        macro_y["inflation"]
        + occ_adj
        + float(rng.normal(0, 0.02))
    )
    if year == 2020:
        base_growth -= macro_y["covid_shock"] * 0.08
    if fraud_persona == "chronic_evader":
        base_growth -= float(rng.uniform(0.01, 0.03))
    return max(0.0, base * (1.0 + base_growth))


def _should_evade(
    persona:          str,
    entry_year:       int,
    year:             int,
    rng:              np.random.Generator,
    one_time_target:  int = -1,
) -> bool:
    yrs = year - entry_year

    if persona == "clean":
        return bool(rng.random() < 0.003)
    elif persona == "chronic_evader":
        return bool(rng.random() < 0.950)
    elif persona == "opportunistic":
        return bool(rng.random() < 0.720)
    elif persona == "late_onset":
        if yrs < 3:
            return bool(rng.random() < 0.10)
        return bool(rng.random() < min(0.82, 0.25 + (yrs - 3) * 0.11))
    elif persona == "escalating":
        return bool(rng.random() < min(0.92, 0.25 + yrs * 0.13))
    elif persona == "one_time":
        return year == one_time_target
    return False


def _validate_zone_year_output(
    df:         pd.DataFrame,
    zone:       int,
    year:       int,
    log,
) -> dict:
    """
    Validate a zone-year parquet before writing.
    Returns a dict of pass/fail counts for the log.
    """
    issues   = []
    warnings = []

    # ── Row count sanity ──────────────────────────────────────────────────────
    if len(df) == 0:
        issues.append("EMPTY dataframe — no records generated")
        return {"zone": zone, "year": year, "issues": issues, "warnings": warnings}

    # ── Fraud rate ────────────────────────────────────────────────────────────
    fraud_rate = float(df["fraud_label"].mean())
    if fraud_rate < 0.10 or fraud_rate > 0.35:
        issues.append(f"fraud_rate={fraud_rate:.4f} outside [0.10, 0.35]")
    else:
        log.info("    ✓ fraud_rate=%.4f", fraud_rate)

    # ── fraud_label=1 with fraud_type=none ────────────────────────────────────
    if "fraud_type" in df.columns:
        contradiction = (
            (df["fraud_label"] == 1) & (df["fraud_type"] == "none")
        ).sum()
        pct = contradiction / max(len(df), 1) * 100
        if contradiction > 0:
            warnings.append(
                f"fraud_label=1/fraud_type=none: {contradiction} rows ({pct:.2f}%)"
            )
        else:
            log.info("    ✓ No fraud_label/fraud_type contradictions")

    # ── W2 wages zero rate ────────────────────────────────────────────────────
    if "w2_wages" in df.columns and "taxpayer_type" in df.columns:
        w2_mask = df["taxpayer_type"].isin(["pure_w2", "w2_with_side_biz"])
        w2_rows = df[w2_mask]
    if len(w2_rows) > 0:
        # NaN = correctly nulled by apply_null_policy (missing employment link)
        # 0.0 = actual zero wage written — this should not happen
        actual_zero = (w2_rows["w2_wages"] == 0).sum()   # exact zero, not NaN
        nulled      = w2_rows["w2_wages"].isna().sum()    # NaN = fixed
        pct_zero    = actual_zero / len(w2_rows) * 100
        pct_nulled  = nulled / len(w2_rows) * 100
        if actual_zero > 0:
            issues.append(
                f"W2 actual zero wages: {actual_zero}/{len(w2_rows)} ({pct_zero:.1f}%)"
            )
        else:
            log.info(
                "    ✓ W2 wages OK — %.1f%% nulled (missing links), 0 actual zeros",
                pct_nulled,
            )

    # ── AGI sanity ────────────────────────────────────────────────────────────
    if "agi" in df.columns:
        agi_median = float(df["agi"].median())
        agi_neg    = (df["agi"] < 0).sum()
        if agi_neg > 0:
            issues.append(f"AGI negative: {agi_neg} rows")
        if agi_median < 5_000:
            issues.append(f"AGI median=${agi_median:,.0f} — suspiciously low")
        else:
            log.info("    ✓ AGI median=$%.0f", agi_median)

    # ── Tax logic: taxable_income <= AGI ─────────────────────────────────────
    if "taxable_income" in df.columns and "agi" in df.columns:
        bad_tax = (df["taxable_income"] > df["agi"]).sum()
        if bad_tax > 0:
            issues.append(f"taxable_income > agi: {bad_tax} rows")
        else:
            log.info("    ✓ taxable_income <= agi: OK")

    # ── COGS <= gross_receipts ────────────────────────────────────────────────
    if "cogs" in df.columns and "gross_receipts" in df.columns:
        both = df[df["cogs"].notna() & df["gross_receipts"].notna()]
        bad_cogs = (both["cogs"] > both["gross_receipts"]).sum()
        if bad_cogs > 5:
            warnings.append(f"cogs > gross_receipts: {bad_cogs} rows")
        else:
            log.info("    ✓ cogs <= gross_receipts: OK")

    # ── Evasion rate sanity ───────────────────────────────────────────────────
    if "evasion_rate" in df.columns:
        er = df["evasion_rate"].dropna()
        if len(er) > 0:
            er_max = float(er.max())
            if er_max > 2.0:
                warnings.append(
                    f"evasion_rate max={er_max:.2f} > 2.0 — check cap"
                )
            else:
                log.info("    ✓ evasion_rate max=%.4f", er_max)

    # ── Critical nulls ────────────────────────────────────────────────────────
    MUST_NOT_BE_NULL = [
        "person_id", "tax_year", "zone", "age",
        "fraud_label", "fraud_type", "agi",
        "total_tax_liability", "taxpayer_type",
        "filing_status", "deduction_taken",
    ]
    for col in MUST_NOT_BE_NULL:
        if col not in df.columns:
            issues.append(f"Missing critical column: {col}")
        elif df[col].isna().any():
            n_null = int(df[col].isna().sum())
            issues.append(f"{col} has {n_null} nulls — must be 0")

    if not issues and not warnings:
        log.info(
            "    ✓ Zone %d Year %d validation PASSED (%d rows)",
            zone, year, len(df),
        )
    else:
        for issue in issues:
            log.error("    ✗ Zone %d Year %d — FAIL: %s", zone, year, issue)
        for warn in warnings:
            log.warning("    ⚠ Zone %d Year %d — WARN: %s", zone, year, warn)

    return {
        "zone":     zone,
        "year":     year,
        "rows":     len(df),
        "issues":   issues,
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core person-year record generator
# ─────────────────────────────────────────────────────────────────────────────

def _gen_person_year(
    person:          pd.Series,
    year:            int,
    emp_person:      pd.DataFrame,
    biz_links_person:pd.DataFrame,
    businesses_idx:  pd.DataFrame,
    dist:            dict,
    rng:             np.random.Generator,
    zone_profiles:   dict,
    macro_shocks:    dict,
    std_deductions:  dict,
    ss_wage_base:    dict,
    income_rules:    dict,
    zone_inc_mult:   dict,
    zone_home_rate:  dict,
    compute_tax_fn,
    prior_se_gross:  float | None = None,
) -> Optional[dict]:

    # ── Activity check ────────────────────────────────────────────────────────
    if person["entry_year"] > year:
        return None
    if person["exit_year"] != 9999 and int(person["exit_year"]) < year:
        return None

    # ── Derived scalars ───────────────────────────────────────────────────────
    age_now  = int(person["age_at_entry"]) + (year - int(person["entry_year"]))
    zone     = int(person["zone"])
    ttype    = person["taxpayer_type"]
    fstatus  = person["filing_status"]
    persona  = person["fraud_persona"]
    zp       = zone_profiles[zone]
    macro_y  = macro_shocks[year]
    sch_c    = dist.get("schedule_c", {})
    housing  = dist["housing"][zone]
    inc_mult = zone_inc_mult[zone]
    home_rt  = zone_home_rate[zone]

    forbid = set(income_rules.get(ttype, {}).get("forbidden", []))

    # ── W-2 wages ─────────────────────────────────────────────────────────────
    w2_wages    = 0.0
    employer_id = None

    if "w2_wages" not in forbid and len(emp_person) > 0:
        active_jobs = emp_person[
            (emp_person["employment_start"] <= year)
            & (
                (emp_person["employment_end"] >= year)
                | (emp_person["employment_end"] == 9999)
            )
        ]
        for _, job in active_jobs.iterrows():
            sal = float(job["base_salary_2019"])
            for y in range(2020, year + 1):
                sal = _wage_growth(sal, y, str(job["occupation"]),
                                   persona, rng, macro_shocks[y])
            if int(job.get("is_part_time", 0)):
                sal *= float(rng.uniform(0.35, 0.65))
            if year == 2020 and rng.random() < 0.12:
                sal *= float(rng.uniform(0.2, 0.8))
            w2_wages += sal
            if employer_id is None:
                employer_id = str(job["business_id"])
        w2_wages = round(max(0.0, w2_wages), 2)

    # ── Schedule C ────────────────────────────────────────────────────────────
    se_gross = se_cogs = se_gross_profit = se_expenses = se_net = 0.0
    sch_c_detail: dict[str, float] = {}
    has_schedule_c = 0

    _IND_P_FALLBACK = {
        "median_gross_receipts":  80_000,
        "gross_receipts_log_std": 0.8,
        "cogs_ratio":             {"mean": 0.30, "std": 0.08},
        "deduction_ratios":       {},
        "cash_intensity":         0.2,
    }
    ind_p = _IND_P_FALLBACK

    if "se_gross_receipts" not in forbid and len(biz_links_person) > 0:
        active_links = biz_links_person[
            (biz_links_person["ownership_start"] <= year)
            & (
                (biz_links_person["ownership_end"] >= year)
                | (biz_links_person["ownership_end"] == 9999)
            )
        ]
        if len(active_links) > 0:
            primary_bid = (
                active_links.sort_values("ownership_start")
                            .iloc[0]["business_id"]
            )
            if primary_bid in businesses_idx.index:
                biz     = businesses_idx.loc[primary_bid]
                ind     = str(biz["industry"])
                yrs_biz = year - int(biz["open_year"])
                mat_f   = min(1.0 + yrs_biz * 0.08, 1.6)

                ind_p = (
                    sch_c.get(ind)
                    or sch_c.get("professional_services")
                    or _IND_P_FALLBACK
                )

                if prior_se_gross is not None and prior_se_gross > 0:
                    growth = macro_y["gdp_growth"] + float(rng.normal(0, 0.08))
                    if year == 2020:
                        growth -= macro_y["covid_shock"] * 0.25
                    se_gross = max(
                        0.0,
                        prior_se_gross * (1.0 + growth) * inc_mult
                        * float(np.exp(rng.normal(0, 0.05)))
                    )
                else:
                    log_rec  = (
                        np.log(max(1.0, ind_p["median_gross_receipts"]))
                        + float(rng.normal(0, ind_p["gross_receipts_log_std"]))
                    )
                    log_rec += np.log(max(0.01, 1.0 + macro_y["gdp_growth"]))
                    if year == 2020:
                        log_rec += np.log(
                            max(0.01, 1.0 - macro_y["covid_shock"] * 0.25)
                        )
                    se_gross = max(0.0, np.exp(log_rec) * mat_f * inc_mult)

                cogs_r = float(np.clip(
                    rng.normal(
                        ind_p["cogs_ratio"]["mean"],
                        ind_p["cogs_ratio"]["std"],
                    ),
                    0.0, 0.95,
                ))
                se_cogs         = se_gross * cogs_r
                se_gross_profit = se_gross - se_cogs

                exp_total = 0.0
                for exp_name, ep in ind_p.get("deduction_ratios", {}).items():
                    r   = max(0.0, float(rng.normal(ep["mean"], ep["std"])))
                    amt = round(se_gross * r, 2)
                    sch_c_detail[f"sch_c_{exp_name}"] = amt
                    exp_total += amt

                se_expenses     = round(exp_total, 2)
                se_net          = round(se_gross_profit - se_expenses, 2)
                se_gross        = round(se_gross, 2)
                se_cogs         = round(se_cogs, 2)
                se_gross_profit = round(se_gross_profit, 2)
                has_schedule_c  = 1

    # ── Gig income ────────────────────────────────────────────────────────────
    gig_gross = gig_expenses = gig_net = 0.0
    has_gig = 0

    if "gig_income" not in forbid:
        gig_prob = {
            "pure_w2":          0.08,
            "gig_only":         0.97,
            "w2_with_side_biz": 0.22,
        }.get(ttype, 0.05)
        if rng.random() < gig_prob * macro_y["gig_growth_factor"]:
            gig_gross    = max(0.0, float(np.exp(rng.normal(9.2, 0.88)) * inc_mult))
            gig_expenses = gig_gross * float(rng.uniform(0.18, 0.42))
            gig_net      = round(gig_gross - gig_expenses, 2)
            gig_gross    = round(gig_gross, 2)
            gig_expenses = round(gig_expenses, 2)
            has_gig      = 1

    # ── Rental income ─────────────────────────────────────────────────────────
    rental_gross = rental_expenses = rental_net = rental_dep = 0.0
    has_rental   = 0
    n_rental_units = 0

    if "rental_income" not in forbid and int(person["has_rental_property"]):
        if rng.random() < housing.get("rental_property_rate", 0.15):
            n_rental_units  = int(rng.choice([1,2,3,4,5], p=[0.60,0.22,0.10,0.05,0.03]))
            rent_monthly    = housing["median_rent_2br"] * float(np.exp(rng.normal(0, 0.22)))
            occupancy       = float(rng.uniform(0.82, 0.98))
            rental_gross    = rent_monthly * 12 * n_rental_units * occupancy
            exp_ratio       = np.clip(float(rng.normal(0.45, 0.12)), 0.20, 0.80)
            rental_expenses = rental_gross * exp_ratio
            home_val_r      = (
                housing["median_home_value"] * n_rental_units
                * float(np.exp(rng.normal(0, 0.18)))
            )
            rental_dep      = (home_val_r * 0.80) / 27.5
            rental_net      = round(rental_gross - rental_expenses, 2)
            rental_gross    = round(rental_gross, 2)
            rental_expenses = round(rental_expenses, 2)
            has_rental      = 1

    # ── Investment income ─────────────────────────────────────────────────────
    dividends = cap_lt = cap_st = interest_inc = 0.0
    has_investments = 0
    lt_frac = 0.60   # default — overwritten if investments exist

    inv_prob = {
        "pure_w2":        0.35, "investor":       0.92,
        "business_owner": 0.65, "multi_biz_owner": 0.78,
        "retired":        0.72,
    }.get(ttype, 0.30)
    if age_now > 45:
        inv_prob = min(inv_prob * 1.4, 0.95)

    if rng.random() < inv_prob:
        port_med     = PORTFOLIO_MEDIANS.get(ttype, _DEFAULT_PORTFOLIO_MEDIAN)
        portfolio    = float(np.exp(rng.normal(np.log(port_med), 0.85)))
        dividends    = portfolio * float(rng.uniform(0.015, 0.025))
        mkt_ret      = MKT_RETURNS.get(year, 0.10)
        real_r       = float(rng.uniform(0.08, 0.35))
        cg_total     = portfolio * max(0.0, mkt_ret) * real_r
        lt_frac      = float(rng.uniform(0.55, 0.85))
        cap_lt       = cg_total * lt_frac
        cap_st       = cg_total * (1.0 - lt_frac)
        fed_proxy    = FED_RATE_PROXY.get(year, 0.02)
        interest_inc = portfolio * 0.25 * max(
            0.0, fed_proxy + float(rng.normal(0, 0.005))
        )
        dividends    = round(dividends, 2)
        cap_lt       = round(cap_lt, 2)
        cap_st       = round(cap_st, 2)
        interest_inc = round(interest_inc, 2)
        has_investments = 1

    # ── Crypto ────────────────────────────────────────────────────────────────
    crypto_proceeds = crypto_basis = crypto_gain = 0.0
    has_crypto     = int(person["is_crypto_user"])
    has_crypto_txn = 0

    if has_crypto and rng.random() < 0.45:
        price_idx       = macro_y["crypto_price_index"]
        position        = float(np.exp(rng.normal(8.5, 1.8)))
        crypto_proceeds = round(position * float(rng.uniform(0.1, 1.0)), 2)
        crypto_basis    = round(
            crypto_proceeds / price_idx * float(rng.uniform(0.5, 2.0)), 2
        )
        crypto_gain     = round(crypto_proceeds - crypto_basis, 2)
        has_crypto_txn  = 1

    # ── Foreign income ────────────────────────────────────────────────────────
    foreign_income = foreign_bal = 0.0
    fbar_required  = 0
    has_foreign    = int(person["has_foreign_account"])

    if has_foreign:
        foreign_bal    = round(float(np.exp(rng.normal(11.5, 1.2))), 2)
        foreign_income = round(foreign_bal * float(rng.uniform(0.02, 0.08)), 2)
        fbar_required  = int(foreign_bal > 10_000)

    # ── Retirement / other income ─────────────────────────────────────────────
    ss_income  = 0.0
    pension    = 0.0
    ira_dist   = 0.0
    unemp_comp = 0.0

    if age_now >= 62 and ttype == "retired":
        ss_income = round(max(0.0, float(rng.normal(18_500, 4_200))), 2)

    if ttype == "retired":
        if rng.random() < 0.55:
            pension  = round(max(0.0, float(np.exp(rng.normal(9.8, 0.7)))), 2)
        if rng.random() < 0.65:
            ira_dist = round(max(0.0, float(np.exp(rng.normal(9.5, 0.9)))), 2)

    if ttype not in ("business_owner", "multi_biz_owner", "investor", "retired"):
        if rng.random() < macro_y["unemployment"] * 0.35:
            unemp_comp = round(float(rng.uniform(8_000, 28_000)), 2)

    # ── K-1 / owner salary ────────────────────────────────────────────────────
    k1_income    = 0.0
    owner_salary = 0.0
    if ttype in ("business_owner", "multi_biz_owner"):
        k1_income    = round(se_net * float(rng.uniform(0.60, 1.20)), 2)
        owner_salary = w2_wages

    # ── Pre-fraud total income ────────────────────────────────────────────────
    total_pre_fraud = round(
        w2_wages + se_net + rental_net + gig_net
        + dividends + cap_lt + cap_st + interest_inc
        + crypto_gain + foreign_income
        + ss_income + pension + ira_dist + unemp_comp
        + k1_income,
        2,
    )

    # ── Utility consumption ───────────────────────────────────────────────────
    ZONE_CLIMATE_FACTOR = {1: 0.90, 2: 0.95, 3: 1.15, 4: 0.80, 5: 1.05}
    heat_f    = ZONE_CLIMATE_FACTOR[zone]
    hhsz_f    = float(rng.uniform(0.9, 1.5))
    base_kwh  = float(rng.normal(10_800, 2_200)) * heat_f * hhsz_f
    inc_adj   = 1.0 + np.log1p(max(0.0, total_pre_fraud) / 60_000) * 0.08
    elec_kwh  = round(max(0.0, base_kwh * inc_adj), 1)
    gas_th    = round(max(0.0, float(rng.normal(580, 180)) * heat_f), 1)
    water_gal = round(max(0.0, float(rng.normal(58_000, 12_000)) * hhsz_f), 1)
    util_cost = round(elec_kwh * 0.12 + gas_th * 1.15 + water_gal * 0.004, 2)
    util_inc_r = round(util_cost / max(total_pre_fraud, 1.0), 4)

    # ── Fraud injection ───────────────────────────────────────────────────────
        # ── Fraud injection ───────────────────────────────────────────────────────
    evading = _should_evade(
        persona,
        int(person["entry_year"]),
        year,
        rng,
        one_time_target=int(person.get("one_time_target_year", -1)),
    )

    # ── Guard: if primary_fraud_type is "none" the person has no fraud
    # mechanism defined — kill the evasion flag rather than write a
    # fraud_label=1 / fraud_type=none contradiction into the parquet.
    _raw_fraud_type = str(person["primary_fraud_type"])
    if evading and _raw_fraud_type == "none":
        evading = False

    fraud_type = _raw_fraud_type if evading else "none"
    fraud_cat  = FRAUD_SCHEMA.get(fraud_type, "none")
    evasion_amt: float = 0.0
    fictitious_ded_amt: float = 0.0

    if evading and fraud_type != "none":

        if fraud_type == "unreported_cash_income":
            rate        = float(rng.uniform(0.05, 0.45))
            evasion_amt = total_pre_fraud * rate * float(rng.uniform(0.3, 1.2))
  
            

        elif fraud_type == "1099_not_reported":
            hidden       = (gig_net + interest_inc) * float(rng.uniform(0.4, 1.0))
            gig_net      = round(max(0.0, gig_net - hidden * 0.6), 2)
            interest_inc = round(max(0.0, interest_inc - hidden * 0.4), 2)
            evasion_amt  = hidden
            

        elif fraud_type == "gig_income_omitted":
            rate        = float(rng.uniform(0.30, 0.95))
            omitted     = gig_gross * rate
            gig_net     = round(max(0.0, gig_net * (1.0 - rate)), 2)
            evasion_amt = omitted
            

        elif fraud_type == "rental_income_hidden":
            rate        = float(rng.uniform(0.25, 0.85))
            hidden      = rental_gross * rate
            rental_net  = round(max(0.0, rental_net - hidden), 2)
            evasion_amt = hidden
            

        elif fraud_type == "farm_income_hidden":
            rate        = float(rng.uniform(0.10, 0.50))
            evasion_amt = total_pre_fraud * rate * 0.3
    # ADD: reduce net income to make it visible
    # Farm income flows through se_net for tax purposes
            farm_hidden = evasion_amt
            se_net      = round(max(0.0, se_net - farm_hidden), 2)

            

        elif fraud_type == "crypto_unreported":
            hidden      = max(0.0, crypto_gain) * float(rng.uniform(0.50, 1.00))
            crypto_gain = round(max(0.0, crypto_gain - hidden), 2)
            evasion_amt = hidden
            

        elif fraud_type == "offshore_hidden":
            hidden         = foreign_income * float(rng.uniform(0.60, 1.00))
            foreign_income = round(max(0.0, foreign_income - hidden), 2)
            evasion_amt    = hidden
            

        elif fraud_type == "fictitious_deductions":
            fake        = (
                float(np.exp(rng.normal(7.8, 0.8)))
                + float(np.exp(rng.normal(7.2, 0.7)))
            )
            evasion_amt = fake
            fictitious_deductions = fake
            

        elif fraud_type == "expense_recharacterization":
            personal_as_biz = float(np.exp(rng.normal(8.5, 0.9)))
            evasion_amt     = personal_as_biz
            
            for k in ("sch_c_meals", "sch_c_car_truck", "sch_c_home_office"):
                if k in sch_c_detail:
                    sch_c_detail[k] = round(sch_c_detail[k] + personal_as_biz * 0.33, 2)

        elif fraud_type == "inflated_cogs":
            rate            = float(rng.uniform(0.05, 0.30))
            inflation       = se_gross * rate
            se_cogs         = round(se_cogs + inflation, 2)
            se_gross_profit = round(se_gross - se_cogs, 2)
            se_net          = round(se_gross_profit - se_expenses, 2)
            evasion_amt     = inflation
            

        elif fraud_type == "low_salary_scorp":
            if w2_wages > 0:
                supp_rate   = float(rng.uniform(0.20, 0.60))
                suppressed  = w2_wages * supp_rate
                w2_wages    = round(w2_wages - suppressed, 2)
                evasion_amt = suppressed
                

        elif fraud_type == "payroll_underreporting":
                off_bks     = float(sch_c_detail.get("sch_c_wages", 0)) * \
                  float(rng.uniform(0.10, 0.40))
                evasion_amt = off_bks
    # ADD: reduce sch_c_wages to make it visible
                if "sch_c_wages" in sch_c_detail:
                    sch_c_detail["sch_c_wages"] = round(
            max(0.0, sch_c_detail["sch_c_wages"] - off_bks), 2
        )
            

        elif fraud_type == "worker_misclassification":
                misclassified = float(sch_c_detail.get("sch_c_wages", 0)) * \
                    float(rng.uniform(0.25, 0.75))
                evasion_amt   = misclassified * 0.0765
    # ADD: move wages from sch_c_wages to se_expenses
    # (reclassified workers show as contractor expenses not wages)
                if "sch_c_wages" in sch_c_detail:
                    sch_c_detail["sch_c_wages"] = round(
            max(0.0, sch_c_detail["sch_c_wages"] - misclassified), 2
        )
            

        elif fraud_type == "revenue_suppression":
            rate            = float(rng.uniform(0.05, 0.30))
            suppressed      = se_gross * rate
            se_gross        = round(max(0.0, se_gross - suppressed), 2)
            se_gross_profit = round(se_gross - se_cogs, 2)
            se_net          = round(se_gross_profit - se_expenses, 2)
            evasion_amt     = suppressed
            

        elif fraud_type == "cash_skimming":
            rate            = float(rng.uniform(0.05, 0.25))
            skimmed         = se_gross * rate
            se_gross        = round(max(0.0, se_gross - skimmed), 2)
            se_gross_profit = round(se_gross - se_cogs, 2)
            se_net          = round(se_gross_profit - se_expenses, 2)
            evasion_amt     = skimmed
        

        elif fraud_type == "shell_company_income_shifting":
            rate        = float(rng.uniform(0.15, 0.55))
            shifted     = max(0.0, se_net) * rate
            se_net      = round(max(0.0, se_net - shifted), 2)
            evasion_amt = shifted
            

        elif fraud_type == "capital_gains_omit":
            if has_investments and (cap_lt + cap_st) > 0:
                omitted     = (cap_lt + cap_st) * float(rng.uniform(0.40, 1.00))
                cap_lt      = round(max(0.0, cap_lt - omitted * lt_frac), 2)
                cap_st      = round(max(0.0, cap_st - omitted * (1.0 - lt_frac)), 2)
                evasion_amt = omitted
                
            else:
                fraud_type  = "fictitious_deductions"
                fake        = (
                    float(np.exp(rng.normal(7.8, 0.8)))
                    + float(np.exp(rng.normal(7.2, 0.7)))
                )
                evasion_amt = fake
                

        evasion_amt = round(max(0.0, evasion_amt), 2)

    # ── Deductions ────────────────────────────────────────────────────────────
    std_ded   = std_deductions.get(year, {}).get(fstatus, 12_950)
    owns_home = (int(person["has_rental_property"]) == 1 or rng.random() < home_rt)
    home_val  = housing["median_home_value"] * float(np.exp(rng.normal(0, 0.30)))
    mort_int  = 0.0
    prop_tax  = 0.0

    if owns_home:
        mrate    = (
            housing.get("mortgage_rates", {}).get(year, 0.045)
            + float(rng.normal(0, 0.004))
        )
        loan_bal = home_val * float(rng.uniform(0.35, 0.85))
        mort_int = loan_bal * mrate
        prop_tax = home_val * float(rng.uniform(0.007, 0.022))

    salt_ded    = min(prop_tax + float(rng.uniform(1_000, 8_000)), 10_000.0)
    charity     = total_pre_fraud * max(0.0, float(rng.beta(1.5, 12)))
    gross_med   = total_pre_fraud * max(0.0, float(rng.exponential(0.04)))
    ded_medical = max(0.0, gross_med - total_pre_fraud * 0.075)

    total_itemized = mort_int + salt_ded + charity + ded_medical

    if evading and fraud_type == "fictitious_deductions":
        total_itemized += fictitious_ded_amt

    uses_itemized   = total_itemized > std_ded
    deduction_taken = round(total_itemized if uses_itemized else std_ded, 2)

    # ── AGI and SE tax ────────────────────────────────────────────────────────
    se_tax_amt = 0.0
    se_tax_ded = 0.0
    if se_net > 400:
        wage_base  = ss_wage_base.get(year, 160_200)
        se_tax_amt = min(se_net, wage_base) * 0.153
        se_tax_ded = round(se_tax_amt / 2.0, 2)
        se_tax_amt = round(se_tax_amt, 2)

    agi = round(max(0.0,
        w2_wages + se_net + rental_net + gig_net
        + dividends + cap_lt + cap_st + interest_inc
        + crypto_gain + foreign_income
        + ss_income * 0.85
        + pension + ira_dist + unemp_comp
        + k1_income - se_tax_ded
    ), 2)

    qbi_ded = 0.0
    if ttype in ("pure_se", "w2_with_side_biz", "business_owner", "multi_biz_owner"):
        qbi     = max(0.0, se_net + k1_income)
        qbi_ded = round(min(qbi * 0.20, agi * 0.20), 2)

    taxable_income = round(max(0.0, agi - deduction_taken - qbi_ded), 2)
    tax_liability  = compute_tax_fn(taxable_income, year, fstatus)

    # ── Credits ───────────────────────────────────────────────────────────────
    n_dep  = 0
    n_kids = min(n_dep, 3)

    eitc          = 0.0
    earned_income = w2_wages + se_net + gig_net
    if (0 < agi < 57_000
            and earned_income > 0
            and ttype not in ("retired", "investor", "business_owner")):
        eitc = min(float(EITC_MAX.get(n_kids, 0)), tax_liability)

    ctc           = min(n_dep * 2_000.0, max(0.0, tax_liability - eitc))
    total_credits = round(eitc + ctc, 2)
    total_tax     = round(max(0.0, tax_liability - total_credits), 2)
    eff_rate      = round(total_tax / max(agi, 1.0), 4)

    # ── Withholding ───────────────────────────────────────────────────────────
    wage_base_y   = ss_wage_base.get(year, 160_200)
    fica_withheld = round(min(w2_wages, wage_base_y) * 0.062, 2)
    medicare_wh   = round(w2_wages * 0.0145, 2)
    fed_withheld  = round(w2_wages * 0.18, 2)
    refund        = round(max(0.0, fed_withheld - total_tax), 2)
    bal_due       = round(max(0.0, total_tax - fed_withheld), 2)



    # ── Detection signals ─────────────────────────────────────────────────────
    life_inc_ratio = round(util_cost / max(agi * 0.12, 1.0), 4)
    bank_dep_ratio = round((agi + evasion_amt) / max(agi, 1.0), 4)
    ded_inc_ratio  = round(deduction_taken / max(agi, 1.0), 4)
    eff_vs_peer    = round(eff_rate - zp["fraud_base_rate"], 4)

    risk_score = float(person["risk_score_base"])
    if evading:
        risk_score += float(rng.uniform(10, 35))
    risk_score += life_inc_ratio * 15
    risk_score  = round(float(np.clip(risk_score + float(rng.normal(0, 3)), 1, 99)), 1)

    # ── PPP loan ──────────────────────────────────────────────────────────────
    ppp_amt  = 0.0
    recv_ppp = False
    if year in (2020, 2021) and ttype in (
        "business_owner", "multi_biz_owner", "w2_with_side_biz", "pure_se"
    ):
        if rng.random() < 0.35:
            recv_ppp = True
            ppp_amt  = round(float(rng.uniform(5_000, 150_000)), 2)

    # ── Assemble row ──────────────────────────────────────────────────────────
    row: dict = {
        "person_id":              person["person_id"],
        "tax_year":               year,
        "zone":                   zone,
        "age":                    age_now,
        "sex":                    person["sex"],
        "education":              person["education"],
        "filing_status":          fstatus,
        "taxpayer_type":          ttype,
        "primary_occupation":     person["primary_occupation"],
        "entry_cohort":           int(person["entry_year"]),
        "first_year_filing":      int(year == int(person["entry_year"])),
        "employer_id":            employer_id,
        "w2_wages": (w2_wages if w2_wages > 0 else None) if "w2_wages" not in forbid else None,
        "federal_withheld":       fed_withheld   if w2_wages > 0 else None,
        "fica_withheld":          fica_withheld  if w2_wages > 0 else None,
        "medicare_withheld":      medicare_wh    if w2_wages > 0 else None,
        "gross_receipts":         se_gross         if has_schedule_c else None,
        "cogs":                   se_cogs          if has_schedule_c else None,
        "gross_profit":           se_gross_profit  if has_schedule_c else None,
        "total_expenses":         se_expenses      if has_schedule_c else None,
        "net_se_income":          se_net           if has_schedule_c else None,
        "has_schedule_c":         has_schedule_c,
        **{k: (v if has_schedule_c else None) for k, v in sch_c_detail.items()},
        "gig_income":             gig_gross    if has_gig else None,
        "gig_expenses":           gig_expenses if has_gig else None,
        "gig_net":                gig_net      if has_gig else None,
        "has_gig":                has_gig,
        "rental_gross":           rental_gross    if has_rental else None,
        "rental_expenses":        rental_expenses if has_rental else None,
        "rental_depreciation":    round(rental_dep, 2) if has_rental else None,
        "rental_net":             rental_net      if has_rental else None,
        "has_rental":             has_rental,
        "n_rental_units":         n_rental_units  if has_rental else None,
        "dividends":              dividends    if has_investments else None,
        "capital_gains_lt":       cap_lt       if has_investments else None,
        "capital_gains_st":       cap_st       if has_investments else None,
        "interest_income":        interest_inc if has_investments else None,
        "has_investments":        has_investments,
        "crypto_proceeds":        crypto_proceeds if has_crypto_txn else None,
        "crypto_cost_basis":      crypto_basis    if has_crypto_txn else None,
        "crypto_net_gain":        crypto_gain     if has_crypto_txn else None,
        "has_crypto":             has_crypto,
        "foreign_income":         foreign_income if has_foreign else None,
        "foreign_account_balance":foreign_bal    if has_foreign else None,
        "fbar_required":          fbar_required  if has_foreign else None,
        "has_foreign_account":    has_foreign,
        "k1_income":              k1_income    if ttype in ("business_owner","multi_biz_owner") else None,
        "owner_salary":           owner_salary if ttype in ("business_owner","multi_biz_owner") else None,
        "social_security_income": ss_income  if age_now >= 62 else None,
        "pension_income":         pension    if pension > 0   else None,
        "ira_distributions":      ira_dist   if ira_dist > 0  else None,
        "unemployment_comp":      unemp_comp if unemp_comp > 0 else None,
        "uses_itemized":          int(uses_itemized),
        "standard_deduction":     None if uses_itemized else std_ded,
        "itemized_total":         round(total_itemized, 2) if uses_itemized else None,
        "itemized_mortgage_int":  round(mort_int, 2)    if uses_itemized else None,
        "itemized_salt":          round(salt_ded, 2)    if uses_itemized else None,
        "itemized_charitable":    round(charity, 2)     if uses_itemized else None,
        "itemized_medical":       round(ded_medical, 2) if uses_itemized else None,
        "deduction_taken":        deduction_taken,
        "qbi_deduction":          qbi_ded if qbi_ded > 0 else None,
        "se_tax_amount":          se_tax_amt if se_tax_amt > 0 else None,
        "se_tax_deduction":       se_tax_ded if se_tax_ded > 0 else None,
        "agi":                    agi,
        "taxable_income":         taxable_income,
        "tax_before_credits":     round(tax_liability, 2),
        "eitc_credit":            round(eitc, 2) if eitc > 0 else None,
        "child_tax_credit":       round(ctc, 2)  if ctc  > 0 else None,
        "total_credits":          total_credits,
        "total_tax_liability":    total_tax,
        "effective_tax_rate":     eff_rate,
        "federal_withheld_total": fed_withheld,
        "refund_amount":          refund  if refund  > 0 else None,
        "balance_due":            bal_due if bal_due > 0 else None,
        "electricity_kwh":        elec_kwh,
        "gas_therms":             gas_th,
        "water_gallons":          water_gal,
        "utility_cost_estimated": util_cost,
        "utility_income_ratio":   util_inc_r,
        "lifestyle_income_ratio": life_inc_ratio,
        "bank_deposit_ratio":     bank_dep_ratio,
        "deduction_income_ratio": ded_inc_ratio,
        "effective_rate_vs_zone": eff_vs_peer,
        "irs_risk_score":         risk_score,
        "received_ppp":           int(recv_ppp),
        "ppp_loan_amount":        ppp_amt if recv_ppp else None,
        "fraud_label":            int(evading),
        "fraud_type":             fraud_type,
        "fraud_category":         fraud_cat,

        
    }

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Modal functions
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=8,
    memory=32768,
    timeout=7200,
)
def generate_zone_year(zone: int, year: int) -> dict:
    """One container per zone-year. Returns stats dict."""
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    import config as cfg
    from config import (
        ZONE_PROFILES, MACRO_SHOCKS, STANDARD_DEDUCTIONS,
        SS_WAGE_BASE, INCOME_STREAM_RULES,
        ZONE_INCOME_MULTIPLIER, ZONE_HOMEOWNERSHIP_RATE,
        PERSONS_CSV, BUSINESSES_CSV, PB_LINKS_CSV, EMP_LINKS_CSV,
        RANDOM_SEED,
    )
    from utils import (
        get_logger, load_distributions,
        write_parquet, apply_null_policy,
        compute_tax_liability,
    )

    log = get_logger(
        f"05_panels_z{zone}_y{year}",
        f"05_panels_z{zone}_{year}.log",
    )

    rng  = np.random.default_rng(RANDOM_SEED + zone * 100 + year)
    dist = load_distributions()

    log.info("Zone %d Year %d — loading reference data...", zone, year)
    persons    = pd.read_csv(PERSONS_CSV)
    businesses = pd.read_csv(BUSINESSES_CSV)
    biz_links  = pd.read_csv(PB_LINKS_CSV)
    emp_links  = pd.read_csv(EMP_LINKS_CSV)

    zone_persons = persons[persons["zone"] == zone].copy()
    zone_persons = zone_persons[
        (zone_persons["entry_year"] <= year)
        & (
            (zone_persons["exit_year"] == 9999)
            | (zone_persons["exit_year"] >= year)
        )
    ].copy()

    log.info("Zone %d Year %d — %d active persons", zone, year, len(zone_persons))

    biz_idx = businesses.set_index("business_id")

    emp_by_person: dict[str, pd.DataFrame] = {
        pid: grp.reset_index(drop=True)
        for pid, grp in emp_links.groupby("person_id")
    }
    biz_by_person: dict[str, pd.DataFrame] = {
        pid: grp.reset_index(drop=True)
        for pid, grp in biz_links.groupby("person_id")
    }

    # ── Prior-year SE income lookup ───────────────────────────────────────────
    prior_se_lookup: dict[str, float] = {}
    if year > 2019:
        prior_path = cfg.PANEL_BZY_DIR / f"zone_{zone}_{year - 1}.parquet"
        if prior_path.exists():
            try:
                prior_df = pd.read_parquet(
                    prior_path,
                    columns=["person_id", "gross_receipts"],
                )
                prior_df = prior_df[prior_df["gross_receipts"].notna()]
                prior_se_lookup = dict(
                    zip(
                        prior_df["person_id"],
                        prior_df["gross_receipts"].astype(float),
                    )
                )
                log.info(
                    "Zone %d Year %d — loaded %d prior-year SE records",
                    zone, year, len(prior_se_lookup),
                )
            except Exception as e:
                log.warning(
                    "Zone %d Year %d — could not load prior parquet: %s",
                    zone, year, e,
                )

    # ── Person loop ───────────────────────────────────────────────────────────
    _empty_df = pd.DataFrame()
    records: list[dict] = []

    for _, person in zone_persons.iterrows():
        pid   = person["person_id"]
        emp_p = emp_by_person.get(pid, _empty_df)
        biz_p = biz_by_person.get(pid, _empty_df)

        rec = _gen_person_year(
            person           = person,
            year             = year,
            emp_person       = emp_p,
            biz_links_person = biz_p,
            businesses_idx   = biz_idx,
            dist             = dist,
            rng              = rng,
            zone_profiles    = ZONE_PROFILES,
            macro_shocks     = MACRO_SHOCKS,
            std_deductions   = STANDARD_DEDUCTIONS,
            ss_wage_base     = SS_WAGE_BASE,
            income_rules     = INCOME_STREAM_RULES,
            zone_inc_mult    = ZONE_INCOME_MULTIPLIER,
            zone_home_rate   = ZONE_HOMEOWNERSHIP_RATE,
            compute_tax_fn   = compute_tax_liability,
            prior_se_gross   = prior_se_lookup.get(pid, None),
        )
        if rec is not None:
            records.append(rec)

    if not records:
        log.warning("Zone %d Year %d: no records generated", zone, year)
        return {"zone": zone, "year": year, "rows": 0, "fraud_rate": 0.0,
                "issues": 0, "warnings": 0}

    df = pd.DataFrame(records)
    df = apply_null_policy(df)

    # ── Validation ────────────────────────────────────────────────────────────
    log.info("Validating Zone %d Year %d output...", zone, year)
    validation = _validate_zone_year_output(df, zone, year, log)

    if validation["issues"]:
        log.error(
            "Zone %d Year %d: %d validation failures — writing anyway but check logs",
            zone, year, len(validation["issues"]),
        )

    zone_noise = float(rng.normal(0, 0.006))
    year_noise = float(rng.normal(0, 0.004))
    if "lifestyle_income_ratio" in df.columns:
        df["lifestyle_income_ratio"] = (
            df["lifestyle_income_ratio"] + zone_noise + year_noise
        ).clip(0, 5)

    cfg.PANEL_BZY_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.PANEL_BZY_DIR / f"zone_{zone}_{year}.parquet"
    write_parquet(df, out)

    fraud_rate = float(df["fraud_label"].mean())
    log.info(
        "Zone %d Year %d: %d records | fraud_rate=%.3f | issues=%d warnings=%d",
        zone, year, len(df), fraud_rate,
        len(validation["issues"]),
        len(validation["warnings"]),
    )

    final_vol.commit()
    logs_vol.commit()

    return {
        "zone":       zone,
        "year":       year,
        "rows":       len(df),
        "fraud_rate": fraud_rate,
        "issues":     len(validation["issues"]),
        "warnings":   len(validation["warnings"]),
    }


@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=4,
    memory=8192,
    timeout=3600,
)
def combine_year(year: int) -> dict:
    """Combine 5 zone parquets into one by-year parquet."""
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"

    import config as cfg
    from config import ZONES
    from utils import write_parquet

    cfg.PANEL_BY_DIR.mkdir(parents=True, exist_ok=True)

    dfs: list[pd.DataFrame] = []
    for z in ZONES:
        p = cfg.PANEL_BZY_DIR / f"zone_{z}_{year}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
        else:
            print(f"  WARNING: missing zone file {p}")

    if not dfs:
        return {"year": year, "rows": 0, "fraud_rate": 0.0}

    combined = pd.concat(dfs, ignore_index=True)
    out      = cfg.PANEL_BY_DIR / f"all_zones_{year}.parquet"
    write_parquet(combined, out)

    final_vol.commit()

    return {
        "year":       year,
        "rows":       len(combined),
        "fraud_rate": float(combined["fraud_label"].mean()),
    }


@app.local_entrypoint()
def main():
    from config import YEARS, ZONES

    total_issues   = 0
    total_warnings = 0

    for year in YEARS:
        print(f"\n{'='*60}")
        print(f"Generating year {year}...")
        print(f"{'='*60}")

        zone_args    = [(z, year) for z in ZONES]
        zone_results = list(generate_zone_year.starmap(zone_args))

        year_rows = 0
        for r in zone_results:
            issues   = r.get("issues", 0)
            warnings = r.get("warnings", 0)
            flag     = "✓" if issues == 0 else "✗"
            print(
                f"  {flag} Zone {r['zone']} Year {r['year']}: "
                f"{r['rows']:>8,} rows  fraud={r.get('fraud_rate', 0):.3f}  "
                f"issues={issues}  warnings={warnings}"
            )
            year_rows      += r["rows"]
            total_issues   += issues
            total_warnings += warnings

        print(f"  Year {year} total: {year_rows:,} rows")

        print(f"Combining year {year}...")
        combine_result = combine_year.remote(year)
        print(
            f"  Combined: {combine_result['rows']:,} rows  "
            f"fraud={combine_result.get('fraud_rate', 0):.3f}"
        )

    print("\n" + "="*60)
    print("ALL YEARS COMPLETE")
    print("="*60)
    print(f"  Total issues:   {total_issues}")
    print(f"  Total warnings: {total_warnings}")
    if total_issues == 0:
        print("  ✓ All zone-year validations passed")
    else:
        print("  ✗ Some validations failed — check logs before running splits")