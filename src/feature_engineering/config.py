# config.py
# ═══════════════════════════════════════════════════════════════════════════════
# SHARED CONFIGURATION — recalibrated from actual data distributions
#
# Key fixes vs previous version:
#   - bank_deposit_ratio clip raised to 550,000 (actual max = 543,988)
#     Previously clipped at 500 which destroyed the entire signal
#     (fraud p50=8.08, clean p50=5.47, but p99=4899, max=543988)
#   - irs_risk_score thresholds verified correct (0-100 scale)
#   - utility_income_ratio removed from SIGNAL_COLS (zero fraud signal)
#   - CLIP_BOUNDS updated to reflect actual data ranges
#   - FIXED_NORM_BOUNDS updated for bank_deposit_ratio true range
#   - flag thresholds recalibrated to actual percentiles
# ═══════════════════════════════════════════════════════════════════════════════

from pathlib import Path
import numpy as np

# ── Root paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"D:\tax_evasion_synthetic_data\scripts_and_outputs\merged_splits")
FE_ROOT   = DATA_ROOT / "featured"

# ── Input folders ─────────────────────────────────────────────────────────────
BY_YEAR       = DATA_ROOT / "split_by_year"
BY_STATE      = DATA_ROOT / "split_by_state"
BY_STATE_YEAR = DATA_ROOT / "split_by_state_by_year"

# ── Intermediate folders (phases 1-8 only) ────────────────────────────────────
CLEAN_DIR      = FE_ROOT / "clean"
ENGINEERED_DIR = FE_ROOT / "engineered"
PEER_DIR       = FE_ROOT / "peer_stats"

# ── Final output ──────────────────────────────────────────────────────────────
ML_READY_DIR           = DATA_ROOT / "ml_ready"
ML_READY_BY_YEAR       = ML_READY_DIR / "by_year"
ML_READY_BY_STATE      = ML_READY_DIR / "by_state"
ML_READY_BY_STATE_YEAR = ML_READY_DIR / "by_state_by_year"

for folder in [
    CLEAN_DIR, ENGINEERED_DIR, PEER_DIR,
    ML_READY_BY_YEAR, ML_READY_BY_STATE, ML_READY_BY_STATE_YEAR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# ── Dataset constants ─────────────────────────────────────────────────────────
STATES      = ["California", "Florida", "Texas", "New York", "Illinois"]
YEARS       = list(range(2019, 2026))
TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
TEST_YEARS  = [2024, 2025]

# ── Fixed categorical vocabularies ───────────────────────────────────────────
VALID_STATES = ["California", "Florida", "Texas", "New York", "Illinois"]
VALID_TAXPAYER_TYPES = [
    "pure_w2", "business_owner", "gig_only", "investor",
    "multi_biz_owner", "pure_se", "retired", "w2_with_side_biz",
]
VALID_AGE_GROUPS   = ["under_25", "25_35", "35_50", "50_65", "over_65"]
VALID_INCOME_BANDS = [
    "under_25k", "25k_50k",   "50k_75k",   "75k_100k",
    "100k_150k", "150k_250k", "250k_500k", "500k_1M", "over_1M", "UNKNOWN",
]

# ── Income bands ──────────────────────────────────────────────────────────────
INCOME_BANDS = [
    0, 25_000, 50_000, 75_000, 100_000,
    150_000, 250_000, 500_000, 1_000_000, float("inf"),
]
INCOME_BAND_LABELS = [
    "under_25k", "25k_50k",   "50k_75k",   "75k_100k",
    "100k_150k", "150k_250k", "250k_500k", "500k_1M", "over_1M",
]

# ── Tax brackets ──────────────────────────────────────────────────────────────
TAX_BRACKETS = [
    (10_000,       0.10),
    (40_000,       0.12),
    (85_000,       0.22),
    (165_000,      0.24),
    (215_000,      0.32),
    (540_000,      0.35),
    (float("inf"), 0.37),
]

# ── Column groups ─────────────────────────────────────────────────────────────
SCH_C_COLS = [
    "sch_c_advertising", "sch_c_car_truck", "sch_c_meals",
    "sch_c_home_office", "sch_c_wages",     "sch_c_depreciation",
]
INCOME_SOURCE_COLS = [
    "w2_wages", "net_se_income", "gig_net",
    "rental_net", "dividends", "capital_gains_lt", "interest_income",
]

# SIGNAL_COLS: columns imputed via group median before feature engineering.
#
# REMOVED: utility_income_ratio — fraud p50=1.2156, clean p50=1.2136,
#          zero discriminative power, imputing it just adds noise
# REMOVED: effective_rate_vs_zone — not validated in diagnostic
# KEPT:    irs_risk_score — fraud p50=86.9, clean p50=45.6, AUC~0.75
# KEPT:    bank_deposit_ratio — extreme right tail is the signal (max=543988)
# KEPT:    federal_withheld — fraud p50=85.8, clean p50=2458.9, strong INVERTED
# KEPT:    lifestyle_income_ratio — weak but present
# KEPT:    deduction_income_ratio — used in multiple downstream features
SIGNAL_COLS = [
    "lifestyle_income_ratio",
    "bank_deposit_ratio",
    "deduction_income_ratio",
    "irs_risk_score",
    "federal_withheld",
]

ZERO_FILL_COLS = [
    "gig_net", "rental_net", "dividends", "capital_gains_lt",
    "interest_income", "gross_receipts",
] + SCH_C_COLS

LEAKAGE_COLS    = ["fraud_type", "fraud_category", "data_source"]
IDENTIFIER_COLS = ["person_id", "employer_id"]

RAW_SIGNAL_COLS = [
    "bank_deposit_ratio_log",    "bank_deposit_ratio_sq_log",
    "flag_bank_deposit_extreme", "flag_bank_deposit_high",
    "flag_irs_high",             "flag_irs_very_high",
    "flag_irs_moderate",         "irs_risk_score_sq",
    "withholding_rate",          "flag_low_withholding",
    "federal_withheld_log",      "zone_risk",
]

# ── Outlier clip bounds ───────────────────────────────────────────────────────
# bank_deposit_ratio: actual max=543,988, p99=4,899
#   Previous clip at 500 destroyed the entire signal — both fraud and
#   clean were collapsed to the same value. Now clip at 550,000 to
#   preserve the full distribution including extreme fraud cases.
#   Log transform in add_raw_signal_features() handles the skew.
#
# irs_risk_score: actual range -1.34 to 103.96, clip to 0-100 is correct
#
# federal_withheld: actual range -623 to 138,262
#   Negative values are data errors — clip at 0.
#   Max 138,262 is legitimate (high-income W2). Keep at 150,000.
#
# utility_income_ratio: actual range -14 to 4,992 with p95=2,634
#   No fraud signal (p50 identical). Still clip for downstream safety.
#
# w2_wages: actual min=-741 (data error), clip at 0
# deduction_taken: actual min=-623 (data error), clip at 0
CLIP_BOUNDS = {
    "agi":                    (0,           10_000_000),
    "taxable_income":         (0,           10_000_000),
    "w2_wages":               (0,            5_000_000),
    "net_se_income":          (-500_000,     5_000_000),
    "gig_net":                (-200_000,     2_000_000),
    "rental_net":             (-200_000,     2_000_000),
    "dividends":              (0,            5_000_000),
    "capital_gains_lt":       (-500_000,     5_000_000),
    "interest_income":        (0,            2_000_000),
    "gross_receipts":         (0,           20_000_000),
    "total_tax_liability":    (0,            5_000_000),
    "deduction_taken":        (0,            5_000_000),
    "effective_tax_rate":     (0,            0.65),
    "lifestyle_income_ratio": (0,            20.0),
    "bank_deposit_ratio":     (-20.0,   550_000.0),   # FIXED: was 0-500
    "utility_income_ratio":   (-15.0,     5_000.0),   # no clip signal, just bounds
    "deduction_income_ratio": (0,             5.0),
    "irs_risk_score":         (0,           100.0),
    "federal_withheld":       (0,           150_000),  # FIXED: was unbounded
    "age":                    (16,           100),
}

# ── Fixed normalization bounds ────────────────────────────────────────────────
# bank_deposit_ratio_log: log1p(550000) = 13.2, so use 0-14
# bank_deposit_ratio: use log scale bounds for norm, not raw
# federal_withheld: fraud p50=85.8 vs clean p50=2458.9 — INVERTED signal
FIXED_NORM_BOUNDS: dict[str, tuple[float, float]] = {
    "lifestyle_income_ratio":     (0.04,      5.08),
    "bank_deposit_ratio":         (-20.0, 550_000.0),
    "bank_deposit_ratio_log":     (0.0,       14.0),   # FIXED: was 0-6.2
    "utility_income_ratio":       (-15.0,   5_000.0),
    "unexplained_wealth_ratio":   (0.00,       5.20),
    "tax_rate_gap":               (-0.15,      0.35),
    "underpayment_proxy":         (0.00,   25_000.0),
    "bracket_rate_ratio":         (0.00,       1.50),
    "net_deduction_rate":         (0.05,       0.88),
    "sch_c_to_revenue_ratio":     (0.10,       1.25),
    "expense_entropy":            (0.50,       4.10),
    "total_flag_count":           (0.00,       8.00),
    "z_deduction_income_ratio":   (-3.50,      3.50),
    "z_lifestyle_income_ratio":   (-3.50,      3.50),
    "z_sch_c_to_revenue_ratio":   (-3.50,      3.50),
    "benford_deviation_expenses": (0.00,       8.50),
    "round_number_ratio":         (0.00,       0.85),
    "cash_t_method_gap_ratio":    (-1.00,    300.00),
    "lifestyle_asset_score":      (0.05,       4.20),
    "irs_risk_score":             (0.00,     100.00),
    "composite_z_score":          (-3.00,      3.00),
    "federal_withheld":           (0.00,  150_000.0),  # FIXED: was 20,000
    "withholding_rate":           (-0.01,      0.30),
    "zone_risk":                  (0.00,       4.00),
    "cash_t_method_gap":          (0.00, 25_000_000.0),
}

# ── Z-score features and weights ──────────────────────────────────────────────
# irs_risk_score weight = 0.40 (fraud p50=86.9 vs clean p50=45.6)
# bank_deposit_ratio weight = 0.25 (extreme right tail signal)
# federal_withheld removed from Z_SCORE_FEATURES — it is INVERTED
#   (fraud has LOWER values) so z-score would go the wrong direction
#   unless explicitly handled; use withholding_block_score instead
Z_SCORE_FEATURES = [
    "lifestyle_income_ratio",
    "bank_deposit_ratio",
    "deduction_income_ratio",
    "irs_risk_score",
    "sch_c_to_revenue_ratio",
    "net_deduction_rate",
    "tax_rate_gap",
    "income_entropy",
    "expense_entropy",
]
Z_SCORE_WEIGHTS = {
    "irs_risk_score":         0.45,
    "bank_deposit_ratio":     0.25,
    "lifestyle_income_ratio": 0.10,
    "deduction_income_ratio": 0.10,
    "tax_rate_gap":           0.07,
    "expense_entropy":        0.03,
}

# ── Benford's Law expected distribution ──────────────────────────────────────
BENFORD_EXPECTED = np.array(
    [0.30103, 0.17609, 0.12494, 0.09691, 0.07918,
     0.06695, 0.05799, 0.05115, 0.04576],
    dtype=np.float32,
)
assert np.all(BENFORD_EXPECTED > 0),               "Benford values must be positive"
assert abs(BENFORD_EXPECTED.sum() - 1.0) < 0.001, "Benford must sum to ~1"

# ── ML constants ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
EPS          = 1e-9
M_WEIGHT     = 10.0

ALWAYS_KEEP = [
    "fraud_label", "tax_year", "state", "taxpayer_type",
    "agi", "composite_z_score",
]

SELECTION_ALWAYS_KEEP = ALWAYS_KEEP + [
    "irs_risk_score",
    "irs_risk_score_sq",
    "z_irs_risk_score",
    "pct_irs_risk_score",
    "irs_block_score",
    "irs_block_score_nl",
    "bank_deposit_ratio",
    "bank_deposit_ratio_log",
    "bank_deposit_ratio_sq_log",
    "z_bank_deposit_ratio",
    "bank_deposit_block_score",
    "flag_irs_high",
    "flag_irs_very_high",
    "flag_irs_moderate",
    "flag_bank_deposit_extreme",
    "flag_bank_deposit_high",
    "flag_low_withholding",
    "withholding_rate",
    "withholding_block_score",
    "federal_withheld_log",
    "zone_risk",
    "network_risk_score",
    "network_block_score",
    "zscore_block_score",
    "master_fraud_propensity",
]

ALWAYS_DROP = LEAKAGE_COLS + IDENTIFIER_COLS + [
    "age_group", "income_band", "risk_tier",
]

print("Config loaded")
print(f"  Data root  : {DATA_ROOT}")
print(f"  Scratch    : {FE_ROOT}")
print(f"  ML ready   : {ML_READY_DIR}")