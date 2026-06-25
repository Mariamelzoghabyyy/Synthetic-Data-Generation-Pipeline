# convert_distributions.py
"""
Convert existing JSON distribution files into the PKL format
that the pipeline expects.

Run once locally:
    python convert_distributions.py

Output: master_distributions.pkl  (in your distributions folder)
"""

import json
import pickle
import numpy as np
from pathlib import Path

DIST_DIR = Path(r"D:\final_version_ofdata_inshallah\distribution_params")

# ── Load all JSON files ───────────────────────────────────────────────────────

def load_json(name: str) -> dict:
    path = DIST_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("Loading JSON files...")
acs_dist   = load_json("acs_distributions.json")
bls_dist   = load_json("bls_distributions.json")
sch_c_dist = load_json("schedule_c_distributions.json")
zillow_dist = load_json("zillow_distributions.json")
hmda_dist  = load_json("hmda_distributions.json")
tax_gap    = load_json("tax_gap_config.json")
all_dist   = load_json("all_distributions.json")

print("  All JSON files loaded.")

# ── State -> Zone mapping ─────────────────────────────────────────────────────

STATE_TO_ZONE = {
    # Zone 1 - Urban Industrial
    "Illinois": 1, "Indiana": 1, "Michigan": 1, "Ohio": 1,
    "Pennsylvania": 1, "Wisconsin": 1, "New York": 1,
    # Zone 2 - Suburban Professional
    "Connecticut": 2, "Delaware": 2, "District of Columbia": 2,
    "Maryland": 2, "Massachusetts": 2, "New Jersey": 2, "Virginia": 2,
    # Zone 3 - Rural Agricultural
    "Arkansas": 3, "Iowa": 3, "Kansas": 3, "Minnesota": 3,
    "Missouri": 3, "Nebraska": 3, "North Dakota": 3, "Oklahoma": 3,
    "South Dakota": 3, "Kentucky": 3, "Mississippi": 3,
    "Alabama": 3, "South Carolina": 3, "West Virginia": 3,
    "Montana": 3, "Maine": 3, "Vermont": 3, "Wyoming": 3,
    "New Hampshire": 3, "Rhode Island": 3,
    # Zone 4 - Tech Innovation Hub
    "California": 4, "Colorado": 4, "Idaho": 4, "Oregon": 4,
    "Utah": 4, "Washington": 4, "Alaska": 4, "Hawaii": 4, "Nevada": 4,
    # Zone 5 - Mixed Border Economy
    "Arizona": 5, "New Mexico": 5, "Texas": 5, "Louisiana": 5,
    "Florida": 5, "Georgia": 5, "North Carolina": 5, "Tennessee": 5,
}

ZONE_LABELS = {
    1: "Urban_Industrial",
    2: "Suburban_Professional",
    3: "Rural_Agricultural",
    4: "Tech_Innovation_Hub",
    5: "Mixed_Border_Economy",
}

# ── Build ACS bundle ──────────────────────────────────────────────────────────

def build_acs(acs: dict) -> dict:
    print("Building ACS bundle...")

    # National wage distribution
    w2_nat = acs.get("w2_income_national", {})
    wage_income = {
        "log_mean": float(w2_nat.get("log_mean", 10.78)),
        "log_std":  float(w2_nat.get("log_std",  0.82)),
        "p10":  float(w2_nat.get("p10",  12000)),
        "p25":  float(w2_nat.get("p25",  26000)),
        "p50":  float(w2_nat.get("p50",  54000)),
        "p75":  float(w2_nat.get("p75",  89000)),
        "p90":  float(w2_nat.get("p90",  145000)),
        "p95":  float(w2_nat.get("p95",  198000)),
        "p99":  float(w2_nat.get("p99",  480000)),
    }

    # Age distribution (aggregate across states if needed)
    age_by_state = acs.get("age_by_state", {})
    if age_by_state:
        means = [v.get("mean", 42.1) for v in age_by_state.values()
                 if isinstance(v, dict)]
        stds  = [v.get("std",  13.8) for v in age_by_state.values()
                 if isinstance(v, dict)]
        age_dist = {
            "mean": float(np.mean(means)) if means else 42.1,
            "std":  float(np.mean(stds))  if stds  else 13.8,
            "min":  18, "max": 80,
        }
    else:
        age_dist = {"mean": 42.1, "std": 13.8, "min": 18, "max": 80}

    # Education (national aggregate from state data)
    edu_by_state = acs.get("education_by_state", {})
    edu_keys = ["no_hs", "hs_diploma", "some_college", "associates",
                "bachelors", "masters", "professional", "doctorate"]
    edu_fallback = {
        "no_hs": 0.09, "hs_diploma": 0.27, "some_college": 0.20,
        "associates": 0.09, "bachelors": 0.22,
        "masters": 0.09, "professional": 0.02, "doctorate": 0.02,
    }
    if edu_by_state:
        edu_agg = {k: [] for k in edu_keys}
        for state_data in edu_by_state.values():
            if isinstance(state_data, dict):
                for k in edu_keys:
                    if k in state_data:
                        edu_agg[k].append(float(state_data[k]))
        edu_dist = {}
        for k in edu_keys:
            if edu_agg[k]:
                edu_dist[k] = round(float(np.mean(edu_agg[k])), 4)
            else:
                edu_dist[k] = edu_fallback[k]
        # Normalise
        total = sum(edu_dist.values())
        edu_dist = {k: round(v / total, 4) for k, v in edu_dist.items()}
    else:
        edu_dist = edu_fallback

    # Filing status proxy from marital_status_by_age
    mar = acs.get("marital_status_by_age", {})
    fs_fallback = {
        "single": 0.41, "married_joint": 0.38,
        "married_separate": 0.03, "head_of_household": 0.14,
        "qualifying_widow": 0.04,
    }
    if mar:
        # Aggregate across age groups
        married = widowed = single = divorced = separated = 0
        n = 0
        for age_grp, vals in mar.items():
            if isinstance(vals, dict):
                married   += float(vals.get("married",   0))
                widowed   += float(vals.get("widowed",   0))
                single    += float(vals.get("never_married", 0))
                divorced  += float(vals.get("divorced",  0))
                separated += float(vals.get("separated", 0))
                n += 1
        if n > 0:
            married /= n; widowed /= n; single /= n
            divorced /= n; separated /= n
            raw_fs = {
                "single":            single + divorced * 0.6,
                "married_joint":     married * 0.92,
                "married_separate":  married * 0.08 + separated,
                "head_of_household": divorced * 0.4 + single * 0.15,
                "qualifying_widow":  widowed,
            }
            total = sum(raw_fs.values())
            if total > 0:
                fs_dist = {k: round(v / total, 4) for k, v in raw_fs.items()}
            else:
                fs_dist = fs_fallback
        else:
            fs_dist = fs_fallback
    else:
        fs_dist = fs_fallback

    # Self-employment income
    se_raw = acs.get("self_employment_income", {})
    se_dist = {
        "log_mean": float(se_raw.get("log_mean", 10.21)),
        "log_std":  float(se_raw.get("log_std",  1.18)),
    }

    # Per-zone income from w2_income_by_state
    w2_by_state = acs.get("w2_income_by_state", {})
    zone_income = {}
    zone_wage_lists = {z: [] for z in range(1, 6)}

    for state, vals in w2_by_state.items():
        zone = STATE_TO_ZONE.get(state)
        if zone and isinstance(vals, dict):
            lm = vals.get("log_mean")
            ls = vals.get("log_std")
            p50 = vals.get("p50")
            if lm:
                zone_wage_lists[zone].append({
                    "log_mean": float(lm),
                    "log_std":  float(ls) if ls else 0.35,
                    "p50":      float(p50) if p50 else np.exp(float(lm)),
                })

    for zone, items in zone_wage_lists.items():
        if items:
            zone_income[zone] = {
                "log_mean": round(float(np.mean([i["log_mean"] for i in items])), 4),
                "log_std":  round(float(np.mean([i["log_std"]  for i in items])), 4),
                "p50":      round(float(np.mean([i["p50"]      for i in items])), 2),
                "n":        len(items),
            }
            print(f"  Zone {zone} wage p50: ${zone_income[zone]['p50']:,.0f}  "
                  f"({len(items)} states)")

    return {
        "age":             age_dist,
        "wage_income":     wage_income,
        "self_emp_income": se_dist,
        "education":       edu_dist,
        "filing_status":   fs_dist,
        "zone_income":     zone_income,
    }


# ── Build BLS bundle ──────────────────────────────────────────────────────────

# Map from your BLS JSON occupation keys to pipeline occupation names
BLS_KEY_MAP = {
    "Management":                          "management",
    "Business and Financial Operations":   "business_finance",
    "Computer and Mathematical":           "computer_math",
    "Architecture and Engineering":        "architecture_eng",
    "Life, Physical, and Social Science":  "life_sciences",
    "Legal":                               "legal",
    "Educational Instruction and Library": "education",
    "Healthcare Practitioners":            "healthcare_pract",
    "Healthcare Support":                  "healthcare_support",
    "Protective Service":                  "protective_service",
    "Food Preparation and Serving":        "food_service",
    "Building and Grounds Cleaning":       "building_grounds",
    "Personal Care and Service":           "personal_care",
    "Sales and Related":                   "sales",
    "Office and Administrative Support":   "office_admin",
    "Farming, Fishing, and Forestry":      "farming_fishing",
    "Construction and Extraction":         "construction_extract",
    "Installation, Maintenance, Repair":   "installation_repair",
    "Production":                          "production",
    "Transportation and Material Moving":  "transportation",
    "Social Science":                      "social_science",
}

GIG_RATES = {
    "management": 0.05, "business_finance": 0.10,
    "computer_math": 0.18, "architecture_eng": 0.08,
    "life_sciences": 0.06, "legal": 0.12,
    "education": 0.07, "healthcare_pract": 0.08,
    "healthcare_support": 0.05, "food_service": 0.15,
    "sales": 0.12, "office_admin": 0.08,
    "construction_extract": 0.20, "installation_repair": 0.15,
    "production": 0.08, "transportation": 0.25,
    "farming_fishing": 0.30, "personal_care": 0.22,
    "protective_service": 0.03, "building_grounds": 0.18,
    "social_science": 0.08,
}

BLS_FALLBACK = {
    "management":           {"p10": 62000,  "p50": 130000, "p90": 295000,
                             "emp_share": 0.052},
    "business_finance":     {"p10": 52000,  "p50": 85000,  "p90": 148000,
                             "emp_share": 0.068},
    "computer_math":        {"p10": 72000,  "p50": 120000, "p90": 185000,
                             "emp_share": 0.035},
    "architecture_eng":     {"p10": 62000,  "p50": 96000,  "p90": 148000,
                             "emp_share": 0.019},
    "life_sciences":        {"p10": 48000,  "p50": 80000,  "p90": 138000,
                             "emp_share": 0.009},
    "legal":                {"p10": 58000,  "p50": 120000, "p90": 285000,
                             "emp_share": 0.008},
    "education":            {"p10": 38000,  "p50": 58000,  "p90": 90000,
                             "emp_share": 0.065},
    "healthcare_pract":     {"p10": 62000,  "p50": 105000, "p90": 198000,
                             "emp_share": 0.065},
    "healthcare_support":   {"p10": 28000,  "p50": 42000,  "p90": 65000,
                             "emp_share": 0.031},
    "food_service":         {"p10": 23000,  "p50": 32000,  "p90": 52000,
                             "emp_share": 0.088},
    "sales":                {"p10": 28000,  "p50": 52000,  "p90": 115000,
                             "emp_share": 0.097},
    "office_admin":         {"p10": 30000,  "p50": 45000,  "p90": 70000,
                             "emp_share": 0.130},
    "construction_extract": {"p10": 36000,  "p50": 62000,  "p90": 105000,
                             "emp_share": 0.059},
    "installation_repair":  {"p10": 38000,  "p50": 64000,  "p90": 106000,
                             "emp_share": 0.038},
    "production":           {"p10": 30000,  "p50": 48000,  "p90": 80000,
                             "emp_share": 0.059},
    "transportation":       {"p10": 28000,  "p50": 48000,  "p90": 82000,
                             "emp_share": 0.065},
    "farming_fishing":      {"p10": 24000,  "p50": 34000,  "p90": 56000,
                             "emp_share": 0.009},
    "personal_care":        {"p10": 24000,  "p50": 36000,  "p90": 60000,
                             "emp_share": 0.026},
    "protective_service":   {"p10": 35000,  "p50": 57000,  "p90": 92000,
                             "emp_share": 0.025},
    "building_grounds":     {"p10": 26000,  "p50": 38000,  "p90": 62000,
                             "emp_share": 0.032},
    "social_science":       {"p10": 42000,  "p50": 73000,  "p90": 128000,
                             "emp_share": 0.005},
}


def build_bls(bls: dict) -> dict:
    print("Building BLS bundle...")
    raw_occs = bls.get("occupations", {})
    occupations = {}

    # Try to map your BLS keys to pipeline names
    for raw_key, raw_vals in raw_occs.items():
        if not isinstance(raw_vals, dict):
            continue

        # Try direct match first, then partial match
        pipeline_name = BLS_KEY_MAP.get(raw_key)
        if pipeline_name is None:
            for bls_label, pname in BLS_KEY_MAP.items():
                if (bls_label.lower() in raw_key.lower() or
                        raw_key.lower() in bls_label.lower()):
                    pipeline_name = pname
                    break

        if pipeline_name is None:
            continue

        # Extract wage fields - try multiple possible key names
        def _get(d, *keys, default=0):
            for k in keys:
                if k in d and d[k] is not None:
                    try:
                        return float(d[k])
                    except (ValueError, TypeError):
                        pass
            return float(default)

        p50  = _get(raw_vals, "median_annual", "a_median", "p50", "median",
                    default=BLS_FALLBACK.get(pipeline_name, {}).get("p50", 50000))
        p10  = _get(raw_vals, "p10_annual", "a_pct10",  "p10",
                    default=p50 * 0.55)
        p25  = _get(raw_vals, "p25_annual", "a_pct25",  "p25",
                    default=p50 * 0.72)
        p75  = _get(raw_vals, "p75_annual", "a_pct75",  "p75",
                    default=p50 * 1.32)
        p90  = _get(raw_vals, "p90_annual", "a_pct90",  "p90",
                    default=p50 * 1.65)
        emp  = _get(raw_vals, "total_employment", "tot_emp", "employment",
                    default=BLS_FALLBACK.get(pipeline_name, {}).get(
                        "emp_share", 0.05) * 1e6)

        anchor   = p50 if p50 > 0 else 50000
        log_mean = float(np.log(max(anchor, 1)))
        if p25 > 0 and p75 > 0:
            log_std = float(
                (np.log(max(p75, 1)) - np.log(max(p25, 1))) / 1.35
            )
        else:
            log_std = 0.35

        occupations[pipeline_name] = {
            "p10":       int(p10),
            "p25":       int(p25),
            "p50":       int(p50),
            "p75":       int(p75),
            "p90":       int(p90),
            "log_mean":  round(log_mean, 4),
            "log_std":   round(max(log_std, 0.15), 4),
            "emp_share": float(emp),   # normalised below
            "gig_rate":  GIG_RATES.get(pipeline_name, 0.10),
        }
        print(f"  BLS {pipeline_name}: p50=${int(p50):,}")

    # Fill missing occupations from fallback
    for name, fb in BLS_FALLBACK.items():
        if name not in occupations:
            print(f"  BLS {name}: not in JSON - using fallback")
            p50 = fb["p50"]
            occupations[name] = {
                "p10":      int(p50 * 0.55),
                "p25":      int(p50 * 0.72),
                "p50":      int(p50),
                "p75":      int(p50 * 1.32),
                "p90":      int(p50 * 1.65),
                "log_mean": round(float(np.log(p50)), 4),
                "log_std":  0.35,
                "emp_share": fb["emp_share"],
                "gig_rate": GIG_RATES.get(name, 0.10),
            }

    # Normalise emp_share to sum to 1
    total_emp = sum(v["emp_share"] for v in occupations.values())
    for name in occupations:
        occupations[name]["emp_share"] = round(
            occupations[name]["emp_share"] / total_emp, 6
        )

    print(f"  BLS occupations: {len(occupations)}")
    return {"occupations": occupations}


# ── Build Schedule C bundle ───────────────────────────────────────────────────

# Map NAICS codes in your JSON to pipeline industry keys
NAICS_TO_INDUSTRY = {
    "722": "food_service",
    "812": "personal_care",
    "541": "professional_services",
    "621": "healthcare_solo",
    "236": "construction",
    "238": "construction",
    "441": "retail_trade",
    "531": "real_estate_rental",
    "811": "construction",
    "484": "transportation_gig",
    "511": "professional_services",
    "523": "professional_services",
    "611": "professional_services",
    "711": "professional_services",
    "999": "retail_trade",
}

SCH_C_FALLBACK = {
    "retail_trade": {
        "naics": "44-45", "median_gross_receipts": 78000,
        "gross_receipts_log_std": 1.42,
        "cogs_ratio": {"mean": 0.62, "std": 0.12},
        "net_profit_margin": {"mean": 0.082, "std": 0.11},
        "cash_intensity": 0.31,
        "deduction_ratios": {
            "advertising":    {"mean": 0.018, "std": 0.015},
            "car_truck":      {"mean": 0.024, "std": 0.018},
            "depreciation":   {"mean": 0.031, "std": 0.022},
            "insurance":      {"mean": 0.012, "std": 0.009},
            "meals":          {"mean": 0.008, "std": 0.007},
            "office_expense": {"mean": 0.014, "std": 0.011},
            "rent":           {"mean": 0.048, "std": 0.031},
            "repairs":        {"mean": 0.011, "std": 0.009},
            "supplies":       {"mean": 0.019, "std": 0.014},
            "utilities":      {"mean": 0.016, "std": 0.012},
            "wages":          {"mean": 0.088, "std": 0.062},
            "home_office":    {"mean": 0.006, "std": 0.008},
        },
    },
    "food_service": {
        "naics": "722", "median_gross_receipts": 95000,
        "gross_receipts_log_std": 1.38,
        "cogs_ratio": {"mean": 0.35, "std": 0.08},
        "net_profit_margin": {"mean": 0.062, "std": 0.09},
        "cash_intensity": 0.52,
        "deduction_ratios": {
            "advertising":    {"mean": 0.022, "std": 0.018},
            "car_truck":      {"mean": 0.015, "std": 0.012},
            "depreciation":   {"mean": 0.028, "std": 0.019},
            "insurance":      {"mean": 0.018, "std": 0.013},
            "meals":          {"mean": 0.004, "std": 0.004},
            "office_expense": {"mean": 0.009, "std": 0.007},
            "rent":           {"mean": 0.085, "std": 0.041},
            "repairs":        {"mean": 0.022, "std": 0.016},
            "supplies":       {"mean": 0.028, "std": 0.018},
            "utilities":      {"mean": 0.038, "std": 0.022},
            "wages":          {"mean": 0.142, "std": 0.081},
            "home_office":    {"mean": 0.001, "std": 0.003},
        },
    },
    "construction": {
        "naics": "23", "median_gross_receipts": 142000,
        "gross_receipts_log_std": 1.51,
        "cogs_ratio": {"mean": 0.58, "std": 0.14},
        "net_profit_margin": {"mean": 0.092, "std": 0.12},
        "cash_intensity": 0.38,
        "deduction_ratios": {
            "advertising":    {"mean": 0.009, "std": 0.009},
            "car_truck":      {"mean": 0.048, "std": 0.031},
            "depreciation":   {"mean": 0.042, "std": 0.028},
            "insurance":      {"mean": 0.028, "std": 0.018},
            "meals":          {"mean": 0.006, "std": 0.006},
            "office_expense": {"mean": 0.008, "std": 0.007},
            "rent":           {"mean": 0.019, "std": 0.018},
            "repairs":        {"mean": 0.031, "std": 0.022},
            "supplies":       {"mean": 0.042, "std": 0.028},
            "utilities":      {"mean": 0.014, "std": 0.011},
            "wages":          {"mean": 0.168, "std": 0.092},
            "home_office":    {"mean": 0.008, "std": 0.009},
        },
    },
    "professional_services": {
        "naics": "54", "median_gross_receipts": 112000,
        "gross_receipts_log_std": 1.29,
        "cogs_ratio": {"mean": 0.12, "std": 0.09},
        "net_profit_margin": {"mean": 0.198, "std": 0.14},
        "cash_intensity": 0.12,
        "deduction_ratios": {
            "advertising":    {"mean": 0.014, "std": 0.012},
            "car_truck":      {"mean": 0.028, "std": 0.021},
            "depreciation":   {"mean": 0.018, "std": 0.014},
            "insurance":      {"mean": 0.022, "std": 0.016},
            "meals":          {"mean": 0.016, "std": 0.013},
            "office_expense": {"mean": 0.028, "std": 0.019},
            "rent":           {"mean": 0.042, "std": 0.028},
            "repairs":        {"mean": 0.006, "std": 0.006},
            "supplies":       {"mean": 0.014, "std": 0.011},
            "utilities":      {"mean": 0.011, "std": 0.009},
            "wages":          {"mean": 0.098, "std": 0.071},
            "home_office":    {"mean": 0.022, "std": 0.018},
        },
    },
    "healthcare_solo": {
        "naics": "621", "median_gross_receipts": 185000,
        "gross_receipts_log_std": 1.18,
        "cogs_ratio": {"mean": 0.18, "std": 0.08},
        "net_profit_margin": {"mean": 0.228, "std": 0.13},
        "cash_intensity": 0.09,
        "deduction_ratios": {
            "advertising":    {"mean": 0.008, "std": 0.008},
            "car_truck":      {"mean": 0.018, "std": 0.014},
            "depreciation":   {"mean": 0.022, "std": 0.016},
            "insurance":      {"mean": 0.048, "std": 0.028},
            "meals":          {"mean": 0.006, "std": 0.006},
            "office_expense": {"mean": 0.021, "std": 0.016},
            "rent":           {"mean": 0.058, "std": 0.031},
            "repairs":        {"mean": 0.008, "std": 0.007},
            "supplies":       {"mean": 0.028, "std": 0.018},
            "utilities":      {"mean": 0.014, "std": 0.011},
            "wages":          {"mean": 0.118, "std": 0.068},
            "home_office":    {"mean": 0.009, "std": 0.010},
        },
    },
    "real_estate_rental": {
        "naics": "531", "median_gross_receipts": 48000,
        "gross_receipts_log_std": 1.38,
        "cogs_ratio": {"mean": 0.08, "std": 0.06},
        "net_profit_margin": {"mean": 0.312, "std": 0.18},
        "cash_intensity": 0.22,
        "deduction_ratios": {
            "advertising":    {"mean": 0.012, "std": 0.011},
            "car_truck":      {"mean": 0.022, "std": 0.018},
            "depreciation":   {"mean": 0.148, "std": 0.062},
            "insurance":      {"mean": 0.038, "std": 0.024},
            "meals":          {"mean": 0.002, "std": 0.003},
            "office_expense": {"mean": 0.008, "std": 0.007},
            "rent":           {"mean": 0.002, "std": 0.003},
            "repairs":        {"mean": 0.062, "std": 0.038},
            "supplies":       {"mean": 0.012, "std": 0.009},
            "utilities":      {"mean": 0.028, "std": 0.019},
            "wages":          {"mean": 0.031, "std": 0.028},
            "home_office":    {"mean": 0.001, "std": 0.002},
        },
    },
    "transportation_gig": {
        "naics": "484", "median_gross_receipts": 52000,
        "gross_receipts_log_std": 0.82,
        "cogs_ratio": {"mean": 0.28, "std": 0.09},
        "net_profit_margin": {"mean": 0.118, "std": 0.09},
        "cash_intensity": 0.08,
        "deduction_ratios": {
            "advertising":    {"mean": 0.002, "std": 0.003},
            "car_truck":      {"mean": 0.198, "std": 0.082},
            "depreciation":   {"mean": 0.048, "std": 0.031},
            "insurance":      {"mean": 0.042, "std": 0.028},
            "meals":          {"mean": 0.008, "std": 0.007},
            "office_expense": {"mean": 0.004, "std": 0.004},
            "rent":           {"mean": 0.002, "std": 0.003},
            "repairs":        {"mean": 0.038, "std": 0.025},
            "supplies":       {"mean": 0.012, "std": 0.009},
            "utilities":      {"mean": 0.006, "std": 0.005},
            "wages":          {"mean": 0.008, "std": 0.012},
            "home_office":    {"mean": 0.012, "std": 0.011},
        },
    },
    "agriculture": {
        "naics": "11", "median_gross_receipts": 68000,
        "gross_receipts_log_std": 1.55,
        "cogs_ratio": {"mean": 0.55, "std": 0.15},
        "net_profit_margin": {"mean": 0.095, "std": 0.14},
        "cash_intensity": 0.62,
        "deduction_ratios": {
            "advertising":    {"mean": 0.004, "std": 0.005},
            "car_truck":      {"mean": 0.058, "std": 0.038},
            "depreciation":   {"mean": 0.072, "std": 0.042},
            "insurance":      {"mean": 0.032, "std": 0.022},
            "meals":          {"mean": 0.003, "std": 0.004},
            "office_expense": {"mean": 0.005, "std": 0.006},
            "rent":           {"mean": 0.028, "std": 0.022},
            "repairs":        {"mean": 0.048, "std": 0.032},
            "supplies":       {"mean": 0.062, "std": 0.038},
            "utilities":      {"mean": 0.022, "std": 0.016},
            "wages":          {"mean": 0.082, "std": 0.055},
            "home_office":    {"mean": 0.004, "std": 0.006},
        },
    },
}


def build_schedule_c(sch_c: dict) -> dict:
    print("Building Schedule C bundle...")

    # Start with fallback so all 8 industries are always present
    schedule_c = {k: dict(v) for k, v in SCH_C_FALLBACK.items()}

    for naics_code, vals in sch_c.items():
        if naics_code.startswith("_"):
            continue
        if not isinstance(vals, dict):
            continue

        industry = NAICS_TO_INDUSTRY.get(naics_code)
        if industry is None:
            continue

        def _get(d, *keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    try:
                        return float(d[k])
                    except (ValueError, TypeError):
                        pass
            return default

        # Update gross receipts if present
        receipts = _get(vals, "median_gross_receipts", "mean_gross_receipts",
                        "avg_gross_receipts", "gross_receipts")
        if receipts and receipts > 0:
            schedule_c[industry]["median_gross_receipts"] = receipts
            print(f"  Sch-C {industry} (NAICS {naics_code}): "
                  f"receipts=${receipts:,.0f}")

        # Update COGS ratio
        cogs = _get(vals, "cogs_ratio", "cost_of_goods_ratio")
        if cogs and 0 < cogs < 1:
            schedule_c[industry]["cogs_ratio"]["mean"] = cogs

        # Update net profit margin
        npm = _get(vals, "net_profit_margin", "profit_margin",
                   "net_income_ratio")
        if npm is not None:
            schedule_c[industry]["net_profit_margin"]["mean"] = npm

        # Update deduction ratios if present
        ded = vals.get("deduction_ratios", {})
        if isinstance(ded, dict):
            for ded_name, ded_val in ded.items():
                if ded_name in schedule_c[industry]["deduction_ratios"]:
                    if isinstance(ded_val, dict):
                        mean = _get(ded_val, "mean", "ratio")
                        std  = _get(ded_val, "std",  "sigma")
                        if mean is not None:
                            schedule_c[industry]["deduction_ratios"][
                                ded_name]["mean"] = mean
                        if std is not None:
                            schedule_c[industry]["deduction_ratios"][
                                ded_name]["std"] = std
                    elif isinstance(ded_val, (int, float)):
                        schedule_c[industry]["deduction_ratios"][
                            ded_name]["mean"] = float(ded_val)

    print(f"  Schedule C industries: {len(schedule_c)}")
    return schedule_c


# ── Build Zillow / Housing bundle ─────────────────────────────────────────────

STATE_TO_ZONE_ZILLOW = {
    "Illinois": 1, "Indiana": 1, "Michigan": 1, "Ohio": 1,
    "Pennsylvania": 1, "Wisconsin": 1, "New York": 1,
    "Connecticut": 2, "Delaware": 2, "District of Columbia": 2,
    "Maryland": 2, "Massachusetts": 2, "New Jersey": 2, "Virginia": 2,
    "Arkansas": 3, "Iowa": 3, "Kansas": 3, "Minnesota": 3,
    "Missouri": 3, "Nebraska": 3, "North Dakota": 3, "Oklahoma": 3,
    "South Dakota": 3, "Kentucky": 3, "Mississippi": 3,
    "Alabama": 3, "South Carolina": 3, "West Virginia": 3,
    "Montana": 3, "Maine": 3, "Vermont": 3, "Wyoming": 3,
    "New Hampshire": 3, "Rhode Island": 3,
    "California": 4, "Colorado": 4, "Idaho": 4, "Oregon": 4,
    "Utah": 4, "Washington": 4, "Alaska": 4, "Hawaii": 4, "Nevada": 4,
    "Arizona": 5, "New Mexico": 5, "Texas": 5, "Louisiana": 5,
    "Florida": 5, "Georgia": 5, "North Carolina": 5, "Tennessee": 5,
}

MORTGAGE_RATES = {
    2019: {"mean": 0.0394, "std": 0.004},
    2020: {"mean": 0.0311, "std": 0.003},
    2021: {"mean": 0.0296, "std": 0.003},
    2022: {"mean": 0.0536, "std": 0.006},
    2023: {"mean": 0.0671, "std": 0.007},
    2024: {"mean": 0.0689, "std": 0.008},
    2025: {"mean": 0.0642, "std": 0.007},
}

HOMEOWNERSHIP  = {1: 0.48, 2: 0.65, 3: 0.72, 4: 0.52, 5: 0.68}
RENTAL_RATE    = {1: 0.52, 2: 0.28, 3: 0.32, 4: 0.45, 5: 0.48}
RENTAL_PROP    = {1: 0.31, 2: 0.22, 3: 0.12, 4: 0.28, 5: 0.18}
RENTAL_YIELD   = {1: 0.042, 2: 0.038, 3: 0.068, 4: 0.032, 5: 0.062}


def build_housing_and_zones(zillow: dict, acs_bundle: dict) -> tuple:
    """Returns (housing_dict, zone_profiles_dict)."""
    print("Building housing and zone profiles from Zillow...")

    # Aggregate home values per zone
    zone_hv_lists = {z: [] for z in range(1, 6)}
    zone_growth_lists = {z: {y: [] for y in range(2019, 2026)}
                         for z in range(1, 6)}

    for state, vals in zillow.items():
        if state.startswith("_"):
            continue
        zone = STATE_TO_ZONE_ZILLOW.get(state)
        if zone is None:
            continue
        if not isinstance(vals, dict):
            continue

        # Current/recent home value
        hv = None
        for key in ["current_value", "median_value", "zhvi",
                    "home_value", "value_2023", "value_2022"]:
            if key in vals and vals[key]:
                try:
                    hv = float(vals[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Try year-keyed values
        if hv is None:
            for year in [2023, 2022, 2024, 2021]:
                yr_key = str(year)
                if yr_key in vals and vals[yr_key]:
                    try:
                        hv = float(vals[yr_key])
                        break
                    except (ValueError, TypeError):
                        pass

        if hv and hv > 0:
            zone_hv_lists[zone].append(hv)

        # Growth rates by year
        growth = vals.get("annual_growth", vals.get("growth_by_year", {}))
        if isinstance(growth, dict):
            for yr_key, g_val in growth.items():
                try:
                    yr = int(yr_key)
                    if 2019 <= yr <= 2025 and g_val is not None:
                        zone_growth_lists[zone][yr].append(float(g_val))
                except (ValueError, TypeError):
                    pass

    # Zone fallback home values
    ZONE_HV_FALLBACK = {
        1: 285_000, 2: 485_000, 3: 128_000, 4: 785_000, 5: 210_000
    }
    ZONE_GROWTH_FALLBACK = {
        1: {2019: 0.048, 2020: 0.038, 2021: 0.142, 2022: 0.085,
            2023: -0.012, 2024: 0.022, 2025: 0.018},
        2: {2019: 0.055, 2020: 0.062, 2021: 0.198, 2022: 0.092,
            2023: -0.028, 2024: 0.031, 2025: 0.025},
        3: {2019: 0.032, 2020: 0.025, 2021: 0.118, 2022: 0.075,
            2023: 0.008,  2024: 0.015, 2025: 0.012},
        4: {2019: 0.072, 2020: 0.048, 2021: 0.245, 2022: 0.105,
            2023: -0.058, 2024: 0.018, 2025: 0.035},
        5: {2019: 0.038, 2020: -0.015, 2021: 0.128, 2022: 0.095,
            2023: 0.022,  2024: 0.028, 2025: 0.020},
    }

    zone_hv     = {}
    zone_growth = {}
    for zone in range(1, 6):
        if zone_hv_lists[zone]:
            zone_hv[zone] = float(np.median(zone_hv_lists[zone]))
            print(f"  Zone {zone} home value: ${zone_hv[zone]:,.0f}  "
                  f"({len(zone_hv_lists[zone])} states)")
        else:
            zone_hv[zone] = ZONE_HV_FALLBACK[zone]
            print(f"  Zone {zone} home value: fallback ${zone_hv[zone]:,.0f}")

        zone_growth[zone] = {}
        for yr in range(2019, 2026):
            glist = zone_growth_lists[zone][yr]
            if glist:
                zone_growth[zone][yr] = round(float(np.median(glist)), 4)
            else:
                zone_growth[zone][yr] = ZONE_GROWTH_FALLBACK[zone].get(yr, 0.03)

    # Build housing dict
    zone_rent = {z: int(zone_hv[z] * RENTAL_YIELD[z] / 12)
                 for z in range(1, 6)}

    housing = {}
    for zone in range(1, 6):
        housing[zone] = {
            "median_home_value":     zone_hv[zone],
            "rental_rate":           RENTAL_RATE[zone],
            "rental_property_rate":  RENTAL_PROP[zone],
            "home_value_growth":     zone_growth[zone],
            "median_rent_2br":       zone_rent[zone],
            "gross_rental_yield":    {"mean": RENTAL_YIELD[zone], "std": 0.012},
            "homeownership_rate":    HOMEOWNERSHIP[zone],
            "mortgage_rate_by_year": MORTGAGE_RATES,
        }

    # Build zone_profiles
    ZONE_STATIC = {
        1: {"label": "Urban_Industrial", "population_share": 0.28,
            "cost_of_living": 1.20, "cash_business_density": 0.38,
            "gig_economy_rate": 0.18, "foreign_account_prevalence": 0.025,
            "crypto_adoption": 0.09, "rental_market_size": 0.31,
            "fraud_base_rate": 0.087, "business_density": 0.31,
            "cash_business_rate": 0.38,
            "industry_mix": {
                "manufacturing": 0.22, "retail_trade": 0.15,
                "healthcare": 0.18, "construction": 0.12,
                "food_service": 0.10, "transportation": 0.08,
                "professional_services": 0.10, "other": 0.05},
            "fraud_prevalence": {
                "unreported_cash_income": 0.045,
                "payroll_underreporting": 0.038,
                "worker_misclassification": 0.055,
                "revenue_suppression": 0.042,
                "cash_skimming": 0.038,
                "fictitious_deductions": 0.015,
                "crypto_unreported": 0.008}},
        2: {"label": "Suburban_Professional", "population_share": 0.22,
            "cost_of_living": 1.40, "cash_business_density": 0.12,
            "gig_economy_rate": 0.12, "foreign_account_prevalence": 0.035,
            "crypto_adoption": 0.14, "rental_market_size": 0.22,
            "fraud_base_rate": 0.074, "business_density": 0.42,
            "cash_business_rate": 0.12,
            "industry_mix": {
                "professional_services": 0.28, "finance_insurance": 0.18,
                "healthcare": 0.20, "real_estate": 0.10,
                "information_tech": 0.12, "other": 0.12},
            "fraud_prevalence": {
                "fictitious_deductions": 0.052,
                "expense_recharacterization": 0.048,
                "low_salary_scorp": 0.042,
                "offshore_hidden": 0.018,
                "shell_company_income_shifting": 0.022,
                "crypto_unreported": 0.035,
                "1099_not_reported": 0.028}},
        3: {"label": "Rural_Agricultural", "population_share": 0.18,
            "cost_of_living": 0.82, "cash_business_density": 0.55,
            "gig_economy_rate": 0.10, "foreign_account_prevalence": 0.005,
            "crypto_adoption": 0.04, "rental_market_size": 0.12,
            "fraud_base_rate": 0.091, "business_density": 0.22,
            "cash_business_rate": 0.55,
            "industry_mix": {
                "agriculture": 0.30, "construction": 0.18,
                "retail_trade": 0.15, "manufacturing": 0.12,
                "healthcare": 0.10, "other": 0.15},
            "fraud_prevalence": {
                "farm_income_hidden": 0.062,
                "unreported_cash_income": 0.055,
                "worker_misclassification": 0.072,
                "payroll_underreporting": 0.048,
                "rental_income_hidden": 0.038,
                "gig_income_omitted": 0.025}},
        4: {"label": "Tech_Innovation_Hub", "population_share": 0.20,
            "cost_of_living": 1.65, "cash_business_density": 0.08,
            "gig_economy_rate": 0.22, "foreign_account_prevalence": 0.045,
            "crypto_adoption": 0.22, "rental_market_size": 0.28,
            "fraud_base_rate": 0.068, "business_density": 0.58,
            "cash_business_rate": 0.08,
            "industry_mix": {
                "information_tech": 0.38, "professional_services": 0.22,
                "finance_insurance": 0.15, "real_estate": 0.08,
                "other": 0.17},
            "fraud_prevalence": {
                "crypto_unreported": 0.085,
                "offshore_hidden": 0.038,
                "shell_company_income_shifting": 0.045,
                "low_salary_scorp": 0.055,
                "1099_not_reported": 0.042,
                "expense_recharacterization": 0.052,
                "gig_income_omitted": 0.038}},
        5: {"label": "Mixed_Border_Economy", "population_share": 0.12,
            "cost_of_living": 0.95, "cash_business_density": 0.62,
            "gig_economy_rate": 0.20, "foreign_account_prevalence": 0.048,
            "crypto_adoption": 0.07, "rental_market_size": 0.18,
            "fraud_base_rate": 0.062, "business_density": 0.35,
            "cash_business_rate": 0.62,
            "industry_mix": {
                "hospitality_tourism": 0.28, "retail_trade": 0.20,
                "transportation": 0.15, "construction": 0.12,
                "food_service": 0.15, "other": 0.10},
            "fraud_prevalence": {
                "unreported_cash_income": 0.072,
                "offshore_hidden": 0.045,
                "worker_misclassification": 0.068,
                "revenue_suppression": 0.055,
                "cash_skimming": 0.062,
                "rental_income_hidden": 0.042}},
    }

    zone_income = acs_bundle.get("zone_income", {})
    nat_log_mean = acs_bundle["wage_income"]["log_mean"]

    zone_profiles = {}
    for zone in range(1, 6):
        p = dict(ZONE_STATIC[zone])
        p["median_home_value"] = zone_hv[zone]
        p["home_value_growth"] = zone_growth[zone]
        p["rental_rate"]       = RENTAL_RATE[zone]

        # Income multiplier from ACS zone data
        zi = zone_income.get(zone, {})
        if zi and "log_mean" in zi:
            p["income_multiplier"] = round(
                float(np.exp(zi["log_mean"] - nat_log_mean)), 4
            )
            p["income_dist"] = {
                "w2":  {"mu": round(zi["log_mean"], 4),
                        "sigma": round(zi.get("log_std", 0.35), 4)},
                "se":  {"mu": round(zi["log_mean"] - 0.20, 4), "sigma": 0.98},
                "biz": {"mu": round(zi["log_mean"] + 0.45, 4), "sigma": 1.10},
            }
        else:
            # Fallback income multipliers
            p["income_multiplier"] = {
                1: 1.15, 2: 1.35, 3: 0.82, 4: 1.55, 5: 0.92
            }[zone]
            p["income_dist"] = {
                "w2":  {"mu": {1: 10.75, 2: 11.35, 3: 10.45,
                               4: 11.65, 5: 10.65}[zone], "sigma": 0.72},
                "se":  {"mu": {1: 10.55, 2: 11.15, 3: 10.25,
                               4: 11.45, 5: 10.45}[zone], "sigma": 0.98},
                "biz": {"mu": {1: 11.20, 2: 11.85, 3: 10.85,
                               4: 12.05, 5: 11.05}[zone], "sigma": 1.10},
            }
        zone_profiles[zone] = p

    return housing, zone_profiles


# ── Macro shocks (hardcoded — not in any JSON) ────────────────────────────────

MACRO_SHOCKS = {
    2019: {"gdp_growth": 0.023, "unemployment": 0.037, "inflation": 0.023,
           "covid_shock": 0.0, "crypto_price_index": 1.0,
           "irs_audit_rate": 0.0045, "gig_growth_factor": 1.00},
    2020: {"gdp_growth": -0.033, "unemployment": 0.081, "inflation": 0.012,
           "covid_shock": 1.0, "crypto_price_index": 3.1,
           "irs_audit_rate": 0.0032, "gig_growth_factor": 1.28},
    2021: {"gdp_growth": 0.057, "unemployment": 0.054, "inflation": 0.047,
           "covid_shock": 0.6, "crypto_price_index": 12.4,
           "irs_audit_rate": 0.0029, "gig_growth_factor": 1.41},
    2022: {"gdp_growth": 0.021, "unemployment": 0.037, "inflation": 0.080,
           "covid_shock": 0.1, "crypto_price_index": 4.8,
           "irs_audit_rate": 0.0038, "gig_growth_factor": 1.35},
    2023: {"gdp_growth": 0.025, "unemployment": 0.037, "inflation": 0.041,
           "covid_shock": 0.0, "crypto_price_index": 6.2,
           "irs_audit_rate": 0.0044, "gig_growth_factor": 1.29},
    2024: {"gdp_growth": 0.028, "unemployment": 0.039, "inflation": 0.029,
           "covid_shock": 0.0, "crypto_price_index": 15.1,
           "irs_audit_rate": 0.0052, "gig_growth_factor": 1.31},
    2025: {"gdp_growth": 0.024, "unemployment": 0.041, "inflation": 0.026,
           "covid_shock": 0.0, "crypto_price_index": 18.3,
           "irs_audit_rate": 0.0058, "gig_growth_factor": 1.27},
}


# ── Assemble and save ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Converting JSON distributions -> master_distributions.pkl")
    print("=" * 60)

    acs_bundle  = build_acs(acs_dist)
    bls_bundle  = build_bls(bls_dist)
    sch_c       = build_schedule_c(sch_c_dist)
    housing, zone_profiles = build_housing_and_zones(zillow_dist, acs_bundle)

    bundle = {
        "acs":           acs_bundle,
        "bls":           bls_bundle,
        "schedule_c":    sch_c,
        "housing":       housing,
        "zone_profiles": zone_profiles,
        "macro_shocks":  MACRO_SHOCKS,
    }

    # Validate required keys
    print("\nValidating bundle structure...")
    assert set(bundle.keys()) == {
        "acs", "bls", "schedule_c", "housing",
        "zone_profiles", "macro_shocks"
    }, "Missing top-level keys"
    assert len(bundle["zone_profiles"]) == 5, "Need 5 zones"
    assert len(bundle["macro_shocks"])  == 7, "Need 7 years"
    assert len(bundle["bls"]["occupations"]) >= 10, "Need >= 10 occupations"
    assert len(bundle["schedule_c"]) == 8, "Need 8 industries"
    for zone in range(1, 6):
        assert zone in bundle["housing"], f"Missing housing zone {zone}"
        assert zone in bundle["zone_profiles"], f"Missing zone_profile {zone}"
    print("  All validation checks passed.")

    # Save PKL
    out_pkl = DIST_DIR / "master_distributions.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(bundle, f)
    size_mb = out_pkl.stat().st_size / 1e6
    print(f"\nSaved -> {out_pkl}  ({size_mb:.1f} MB)")

    # Save summary JSON
    out_json = DIST_DIR / "distributions_summary.json"
    summary = {
        "acs_wage_p50":       acs_bundle["wage_income"]["p50"],
        "acs_wage_p90":       acs_bundle["wage_income"]["p90"],
        "bls_occupations":    list(bls_bundle["occupations"].keys()),
        "schedule_c_sectors": list(sch_c.keys()),
        "zone_home_values":   {str(z): housing[z]["median_home_value"]
                               for z in range(1, 6)},
        "zone_multipliers":   {str(z): zone_profiles[z]["income_multiplier"]
                               for z in range(1, 6)},
        "macro_years":        list(MACRO_SHOCKS.keys()),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved -> {out_json}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  ACS wage p50:       ${acs_bundle['wage_income']['p50']:>10,.0f}")
    print(f"  ACS wage p90:       ${acs_bundle['wage_income']['p90']:>10,.0f}")
    print(f"  BLS occupations:    {len(bls_bundle['occupations']):>10}")
    print(f"  Schedule C sectors: {len(sch_c):>10}")
    for zone in range(1, 6):
        print(f"  Zone {zone}: home=${housing[zone]['median_home_value']:>9,.0f}  "
              f"income_mult={zone_profiles[zone]['income_multiplier']:.3f}")
    print("=" * 60)
    print("Done. You can now run scripts 02-04 locally and upload "
          "master_distributions.pkl to Modal.")


if __name__ == "__main__":
    main()