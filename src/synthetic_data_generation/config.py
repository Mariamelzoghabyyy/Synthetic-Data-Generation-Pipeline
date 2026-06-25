# config.py
"""
Central configuration for all paths, volume names, and constants.
Detects Modal vs local execution automatically.
"""

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
# Must be the very first executable lines so the encoding is set before
# any logger, print(), or other output call runs.
# On Modal (Linux/utf-8) this is a no-op.
import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from pathlib import Path

# ── Execution environment ─────────────────────────────────────────────────────
IS_MODAL = bool(os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_MODE"))

# ─────────────────────────────────────────────────────────────────────────────
# Base mount points
# Each key in VOLUME_NAMES has a corresponding Path below.
# On Modal these are volume mount paths; locally they are relative dirs.
# ─────────────────────────────────────────────────────────────────────────────
if IS_MODAL:
    RAW_BASE         = Path("/raw_reference")       # volume: tax-raw-reference
    DIST_BASE        = Path("/distributions")       # volume: tax-distributions
    SEED_PANELS_BASE = Path("/seed_panels")         # volume: tax-seed-panels
    GAN_OUTPUT_BASE  = Path("/gan_output")          # volume: tax-gan-output
    MERGED_BASE      = Path("/merged")              # volume: tax-merged
    FINAL_BASE       = Path("/final_dataset")       # volume: tax-final-dataset
    LOGS_BASE        = Path("/logs")                # volume: tax-logs
else:
    RAW_BASE         = Path("raw_reference_data")
    DIST_BASE        = Path("distributions")
    SEED_PANELS_BASE = Path("seed_panels")
    GAN_OUTPUT_BASE  = Path("gan_output")
    MERGED_BASE      = Path("merged")
    FINAL_BASE       = Path("final_dataset")
    LOGS_BASE        = Path("logs")

# ─────────────────────────────────────────────────────────────────────────────
# Raw reference data (inputs — read-only at generation time)
# ─────────────────────────────────────────────────────────────────────────────
RAW_ACS     = RAW_BASE / "acs_pums"
RAW_BLS     = RAW_BASE / "bls_oes" / "national_M2023_dl.xlsx"
RAW_SCHED_C = RAW_BASE / "schedule_c"
RAW_ZILLOW  = RAW_BASE / "zillow" / "State_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
RAW_HMDA    = RAW_BASE / "hmda"

# ─────────────────────────────────────────────────────────────────────────────
# Distributions  (built from raw data; loaded at generation time)
# master_distributions.pkl lives here — this is what Modal has uploaded
# ─────────────────────────────────────────────────────────────────────────────
DIST_DIR  = DIST_BASE                                    # /distributions  (Modal)
DIST_PKL  = DIST_BASE / "master_distributions.pkl"      # the uploaded file
DIST_JSON = DIST_BASE / "distributions_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
# Reference seeds  (intermediate CSVs produced before panel generation)
# Kept inside FINAL_BASE so they travel with the dataset.
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_DIR  = FINAL_BASE / "reference"
PERSONS_CSV    = REFERENCE_DIR / "persons.csv"
BUSINESSES_CSV = REFERENCE_DIR / "businesses.csv"
PB_LINKS_CSV   = REFERENCE_DIR / "person_business_links.csv"
EMP_LINKS_CSV  = REFERENCE_DIR / "employment_links.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Seed panels  (GAN inputs)
# ─────────────────────────────────────────────────────────────────────────────
SEED_DIR       = SEED_PANELS_BASE / "individuals"
SEED_W2        = SEED_DIR / "seed_w2_employees.parquet"
SEED_SE        = SEED_DIR / "seed_self_employed.parquet"
SEED_ITEMIZERS = SEED_DIR / "seed_itemizers.parquet"
SEED_COMPLIANT = SEED_DIR / "seed_panel_compliant.parquet"
SEED_EVADERS   = SEED_DIR / "seed_panel_evaders.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# GAN outputs
# ─────────────────────────────────────────────────────────────────────────────
GAN_OUTPUT_DIR    = GAN_OUTPUT_BASE / "individuals"
GAN_W2            = GAN_OUTPUT_DIR / "ctab_w2_generated.parquet"
GAN_SE            = GAN_OUTPUT_DIR / "ctab_se_generated.parquet"
GAN_ITEMIZERS     = GAN_OUTPUT_DIR / "ctab_itemizers_generated.parquet"
GAN_COMPLIANT_SEQ = GAN_OUTPUT_DIR / "timegan_compliant_sequences.parquet"
GAN_EVADER_SEQ    = GAN_OUTPUT_DIR / "timegan_evader_sequences.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# Merged output
# ─────────────────────────────────────────────────────────────────────────────
MERGED_DIR  = MERGED_BASE / "individuals"
MERGED_FULL = MERGED_DIR  / "merged_full.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# Panel outputs  (zone × year slices + flat by-year files)
# ─────────────────────────────────────────────────────────────────────────────
PANEL_BZY_DIR = FINAL_BASE / "individuals" / "by_zone_by_year"
PANEL_BY_DIR  = FINAL_BASE / "individuals" / "by_year"

# ─────────────────────────────────────────────────────────────────────────────
# Train / val / test splits
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_DIR = FINAL_BASE / "individuals" / "train"
VAL_DIR   = FINAL_BASE / "individuals" / "val"
TEST_DIR  = FINAL_BASE / "individuals" / "test"

TRAIN_ALL       = TRAIN_DIR / "train_all.parquet"
TRAIN_EMPLOYEES = TRAIN_DIR / "train_employees.parquet"
TRAIN_SE        = TRAIN_DIR / "train_self_employed.parquet"
TRAIN_ITEMIZERS = TRAIN_DIR / "train_itemizers.parquet"
VAL_2024        = VAL_DIR   / "val_2024.parquet"
TEST_2025       = TEST_DIR  / "test_2025.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_DIR = FINAL_BASE / "schema"

# ─────────────────────────────────────────────────────────────────────────────
# Modal volume names  (must match the names used in your modal app definition)
# ─────────────────────────────────────────────────────────────────────────────
VOLUME_NAMES = {
    "raw":       "taxxx-raw-reference",   # mounts at RAW_BASE
    "dists":     "taxxx",   # mounts at DIST_BASE
    "panels":    "taxxx-seed-panels",     # mounts at SEED_PANELS_BASE
    "gan_out":   "taxxx-gan-output",      # mounts at GAN_OUTPUT_BASE
    "merged":    "taxxx-merged",          # mounts at MERGED_BASE
    "final":     "taxxx-final-dataset",   # mounts at FINAL_BASE
    "logs":      "taxxx-logs",            # mounts at LOGS_BASE
}
# NOTE: "reference" seeds live inside FINAL_BASE ("tax-final-dataset"),
# so no separate volume is needed for them.

# ─────────────────────────────────────────────────────────────────────────────
# Dataset constants
# ─────────────────────────────────────────────────────────────────────────────
YEARS = list(range(2019, 2026))   # 2019-2025 inclusive (7 years)
ZONES = [1, 2, 3, 4, 5]

# N_PERSONS_BASE  : unique synthetic persons generated before augmentation
# N_PERSONS_TOTAL : after GAN augmentation / oversampling
# N_BIZ_TOTAL     : unique synthetic businesses
N_PERSONS_BASE  = 360_000
N_PERSONS_TOTAL = 460_000
N_BIZ_TOTAL     = 110_000

RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Fraud-rate targets
#
# Blended check (approximate population weights):
#   pure_w2(0.32) + w2_side_biz(0.12) => ~44 % W2-ish  @ 0.20 => 0.088
#   pure_se(0.14) + gig(0.10)         => ~24 % SE-ish   @ 0.28 => 0.067
#   business_owner(0.10)+multi(0.06)  => ~16 % biz      @ 0.28 => 0.045
#   retired(0.08) + investor(0.08)    => ~16 % other     @ 0.15 => 0.024
#   weighted overall ≈ 0.224  →  0.21 is a reasonable (slightly conservative)
#   overall target once non-itemizer retired/investor lower-fraud groups
#   pull the rate down.  FRAUD_RATE_OVERALL is the enforced dataset target.
# ─────────────────────────────────────────────────────────────────────────────
FRAUD_RATE_OVERALL   = 0.21
FRAUD_RATE_W2        = 0.20   # pure W2 — lowest (third-party reporting)
FRAUD_RATE_SE        = 0.28   # self-employed — highest (cash / under-reporting)
FRAUD_RATE_ITEMIZERS = 0.18   # itemizers only — deduction inflation risk
FRAUD_RATE_TOLERANCE = 0.015  # acceptable +/- band around each target

# ─────────────────────────────────────────────────────────────────────────────
# Zone profiles
# population_share  : fraction of total synthetic population in this zone
# fraud_base_rate   : baseline fraud probability before individual modifiers
# industry_mix      : shares must sum to 1.0 within each zone
# cash_business_density : fraction of businesses that are cash-heavy
# ─────────────────────────────────────────────────────────────────────────────
ZONE_PROFILES = {
    1: {  # Dense urban / low-income — high cash, high fraud
        "population_share":          0.28,
        "fraud_base_rate":           0.08,
        "foreign_account_prevalence": 0.12,
        "crypto_adoption":           0.18,
        "rental_market_size":        0.22,
        "industry_mix": {
            "retail":                0.22,
            "food_service":          0.18,
            "construction":          0.15,
            "professional_services": 0.20,
            "healthcare":            0.08,
            "real_estate":           0.07,
            "tech":                  0.10,
        },  # sum = 1.00
        "cash_business_density":     0.35,
    },
    2: {  # Suburban / mixed — moderate
        "population_share":          0.22,
        "fraud_base_rate":           0.07,
        "foreign_account_prevalence": 0.10,
        "crypto_adoption":           0.20,
        "rental_market_size":        0.18,
        "industry_mix": {
            "retail":                0.20,
            "food_service":          0.20,
            "construction":          0.12,
            "professional_services": 0.22,
            "healthcare":            0.10,
            "real_estate":           0.08,
            "tech":                  0.08,
        },  # sum = 1.00
        "cash_business_density":     0.30,
    },
    3: {  # Rural / small-town — construction/retail heavy
        "population_share":          0.18,
        "fraud_base_rate":           0.06,
        "foreign_account_prevalence": 0.08,
        "crypto_adoption":           0.16,
        "rental_market_size":        0.15,
        "industry_mix": {
            "retail":                0.25,
            "food_service":          0.20,
            "construction":          0.18,
            "professional_services": 0.18,
            "healthcare":            0.07,
            "real_estate":           0.07,
            "tech":                  0.05,
        },  # sum = 1.00
        "cash_business_density":     0.25,
    },
    4: {  # Affluent suburban / tech corridor
        "population_share":          0.18,
        "fraud_base_rate":           0.05,
        "foreign_account_prevalence": 0.14,
        "crypto_adoption":           0.28,
        "rental_market_size":        0.10,
        "industry_mix": {
            "retail":                0.15,
            "food_service":          0.10,
            "construction":          0.08,
            "professional_services": 0.22,
            "healthcare":            0.10,
            "real_estate":           0.10,
            "tech":                  0.25,
        },  # sum = 1.00
        "cash_business_density":     0.20,
    },
    5: {  # High-net-worth / international — offshore risk
        "population_share":          0.14,
        "fraud_base_rate":           0.04,
        "foreign_account_prevalence": 0.20,
        "crypto_adoption":           0.35,
        "rental_market_size":        0.08,
        "industry_mix": {
            "retail":                0.10,
            "food_service":          0.10,
            "construction":          0.05,
            "professional_services": 0.25,
            "healthcare":            0.15,
            "real_estate":           0.15,
            "tech":                  0.20,
        },  # sum = 1.00
        "cash_business_density":     0.15,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Taxpayer type weights  (must sum to 1.0)
# Used to assign taxpayer_type during person generation.
# sum = 0.32+0.14+0.12+0.10+0.06+0.10+0.08+0.08 = 1.00
# ─────────────────────────────────────────────────────────────────────────────
TAXPAYER_TYPES = {
    "pure_w2":        0.32,
    "pure_se":        0.14,
    "w2_with_side_biz": 0.12,
    "business_owner": 0.10,
    "multi_biz_owner": 0.06,
    "gig_only":       0.10,
    "retired":        0.08,
    "investor":       0.08,
}

# ── Standard deductions by year and filing status ─────────────────────────────
# Source: IRS Rev. Proc. for each tax year
STANDARD_DEDUCTIONS: dict[int, dict[str, float]] = {
    2019: {
        "single":            12_200,
        "married_joint":     24_400,
        "married_separate":  12_200,
        "head_of_household": 18_350,
        "qualifying_widow":  24_400,
    },
    2020: {
        "single":            12_400,
        "married_joint":     24_800,
        "married_separate":  12_400,
        "head_of_household": 18_650,
        "qualifying_widow":  24_800,
    },
    2021: {
        "single":            12_550,
        "married_joint":     25_100,
        "married_separate":  12_550,
        "head_of_household": 18_800,
        "qualifying_widow":  25_100,
    },
    2022: {
        "single":            12_950,
        "married_joint":     25_900,
        "married_separate":  12_950,
        "head_of_household": 19_400,
        "qualifying_widow":  25_900,
    },
    2023: {
        "single":            13_850,
        "married_joint":     27_700,
        "married_separate":  13_850,
        "head_of_household": 20_800,
        "qualifying_widow":  27_700,
    },
    2024: {
        "single":            14_600,
        "married_joint":     29_200,
        "married_separate":  14_600,
        "head_of_household": 21_900,
        "qualifying_widow":  29_200,
    },
    2025: {
        "single":            15_000,
        "married_joint":     30_000,
        "married_separate":  15_000,
        "head_of_household": 22_500,
        "qualifying_widow":  30_000,
    },
}

# ── Social Security wage base by year ─────────────────────────────────────────
# Source: SSA annual announcements
SS_WAGE_BASE: dict[int, float] = {
    2019: 132_900,
    2020: 137_700,
    2021: 142_800,
    2022: 147_000,
    2023: 160_200,
    2024: 168_600,
    2025: 176_100,
}

# ── Macro-economic shocks by year ─────────────────────────────────────────────
# inflation          : CPI YoY approximate
# gdp_growth         : real GDP growth rate
# covid_shock        : severity multiplier 0-1 (1.0 = peak shock year 2020)
# unemployment       : annual average unemployment rate
# gig_growth_factor  : multiplier on gig participation base probability
# crypto_price_index : crypto price relative to 2019 baseline of 1.0
MACRO_SHOCKS: dict[int, dict] = {
    2019: {
        "inflation":          0.023,
        "gdp_growth":         0.023,
        "covid_shock":        0.000,
        "unemployment":       0.035,
        "gig_growth_factor":  1.000,
        "crypto_price_index": 1.000,
    },
    2020: {
        "inflation":          0.012,
        "gdp_growth":        -0.033,
        "covid_shock":        1.000,
        "unemployment":       0.081,
        "gig_growth_factor":  1.350,
        "crypto_price_index": 3.100,
    },
    2021: {
        "inflation":          0.047,
        "gdp_growth":         0.057,
        "covid_shock":        0.300,
        "unemployment":       0.054,
        "gig_growth_factor":  1.280,
        "crypto_price_index": 8.200,
    },
    2022: {
        "inflation":          0.080,
        "gdp_growth":         0.021,
        "covid_shock":        0.050,
        "unemployment":       0.037,
        "gig_growth_factor":  1.180,
        "crypto_price_index": 3.500,
    },
    2023: {
        "inflation":          0.041,
        "gdp_growth":         0.025,
        "covid_shock":        0.000,
        "unemployment":       0.037,
        "gig_growth_factor":  1.120,
        "crypto_price_index": 4.800,
    },
    2024: {
        "inflation":          0.030,
        "gdp_growth":         0.027,
        "covid_shock":        0.000,
        "unemployment":       0.040,
        "gig_growth_factor":  1.080,
        "crypto_price_index": 9.200,
    },
    2025: {
        "inflation":          0.026,
        "gdp_growth":         0.022,
        "covid_shock":        0.000,
        "unemployment":       0.042,
        "gig_growth_factor":  1.050,
        "crypto_price_index": 7.400,
    },
}

# ── Income stream rules by taxpayer type ──────────────────────────────────────
# forbidden: list of income stream keys that cannot exist for this type.
# Keys must match the forbid checks in _gen_person_year() in script 05.
INCOME_STREAM_RULES: dict[str, dict] = {
    "pure_w2": {
        "forbidden": ["se_gross_receipts", "gig_income"],
    },
    "pure_se": {
        "forbidden": ["w2_wages"],
    },
    "w2_with_side_biz": {
        "forbidden": [],
    },
    "business_owner": {
        "forbidden": [],
    },
    "multi_biz_owner": {
        "forbidden": [],
    },
    "gig_only": {
        "forbidden": ["w2_wages", "se_gross_receipts"],
    },
    "retired": {
        "forbidden": ["w2_wages", "se_gross_receipts", "gig_income"],
    },
    "investor": {
        "forbidden": ["se_gross_receipts", "gig_income"],
    },
}

# ── Zone-level income multipliers ─────────────────────────────────────────────
# Applied to gross receipts and gig income to reflect local income levels.
# Zone 1 = dense urban / lower income; Zone 5 = high-net-worth.
ZONE_INCOME_MULTIPLIER: dict[int, float] = {
    1: 0.85,
    2: 0.95,
    3: 0.90,
    4: 1.15,
    5: 1.40,
}

# ── Zone-level homeownership rates ────────────────────────────────────────────
# Used to determine mortgage interest / property tax deduction eligibility.
ZONE_HOMEOWNERSHIP_RATE: dict[int, float] = {
    1: 0.42,
    2: 0.58,
    3: 0.65,
    4: 0.72,
    5: 0.80,
}

# ── Federal income tax brackets ───────────────────────────────────────────────
# Each entry: (marginal_rate, upper_bound_inclusive).
# Last bound is float("inf").
# Single filer brackets; MFJ doubles each threshold in compute_tax_liability().
# Values approximate — simplified for synthetic data generation.
TAX_BRACKETS: dict[int, list[tuple[float, float]]] = {
    2019: [
        (0.10,   9_700),
        (0.12,  39_475),
        (0.22,  84_200),
        (0.24, 160_725),
        (0.32, 204_100),
        (0.35, 510_300),
        (0.37, float("inf")),
    ],
    2020: [
        (0.10,   9_875),
        (0.12,  40_125),
        (0.22,  85_525),
        (0.24, 163_300),
        (0.32, 207_350),
        (0.35, 518_400),
        (0.37, float("inf")),
    ],
    2021: [
        (0.10,   9_950),
        (0.12,  40_525),
        (0.22,  86_375),
        (0.24, 164_925),
        (0.32, 209_425),
        (0.35, 523_600),
        (0.37, float("inf")),
    ],
    2022: [
        (0.10,  10_275),
        (0.12,  41_775),
        (0.22,  89_075),
        (0.24, 170_050),
        (0.32, 215_950),
        (0.35, 539_900),
        (0.37, float("inf")),
    ],
    2023: [
        (0.10,  11_000),
        (0.12,  44_725),
        (0.22,  95_375),
        (0.24, 182_950),
        (0.32, 231_250),
        (0.35, 578_125),
        (0.37, float("inf")),
    ],
    2024: [
        (0.10,  11_600),
        (0.12,  47_150),
        (0.22, 100_525),
        (0.24, 191_950),
        (0.32, 243_725),
        (0.35, 609_350),
        (0.37, float("inf")),
    ],
    2025: [
        (0.10,  12_000),
        (0.12,  48_475),
        (0.22, 103_350),
        (0.24, 197_300),
        (0.32, 250_525),
        (0.35, 626_350),
        (0.37, float("inf")),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Modal GPU spec
# ─────────────────────────────────────────────────────────────────────────────
GPU_SPEC = "T4"