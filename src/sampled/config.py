from typing import Dict, List

COLS = {
    "person_id":      "person_id",
    "year":           "tax_year",
    "fraud_label":    "fraud_label",
    "fraud_type":     "fraud_type",
    "fraud_category": "fraud_category",
    "state":          "state",
    "taxpayer_type":  "taxpayer_type",
    "first_year":     "first_year_filing",
    "entry_cohort":   "entry_cohort",
}

COLS_TO_DROP: List[str] = ["data_source"]

YEARS: List[int] = list(range(2019, 2026))

TARGET_ROWS_PER_STATE: int = 10_000

# ── Fraud rates from explore.py output ────────────────────────────────
SOURCE_FRAUD_RATES: Dict[str, float] = {
    "california": 0.2731,
    "illinois":   0.1528,
    "new_york":   0.1739,
    "florida":    0.2150,
    "texas":      0.1931,
}

# ── Target fraud rates after downsampling ─────────────────────────────
# Kept realistic but reduced from source
# Each state retains its RELATIVE ordering (CA highest, IL lowest)
TARGET_FRAUD_RATES: Dict[str, float] = {
    "california": 0.22, # High end of your range
    "illinois":   0.16, # Low end of your range
    "new_york":   0.18,
    "florida":    0.21,
    "texas":      0.19,
}

# ── Lifecycle bucket proportions to PRESERVE in sample ────────────────
# Derived from explore.py — approximate source proportions
# These are PERSON-level proportions, not row-level
# Format: (min_years, max_years_inclusive)
LIFECYCLE_BUCKETS = {
    "transient":   (1, 1),
    "short":       (2, 3),
    "mid":         (4, 5),
    "persistent":  (6, 7),
}

# Target person-level proportion per lifecycle bucket
# Set to approximately match source data
# Must sum to 1.0
LIFECYCLE_TARGET_PROPORTIONS: Dict[str, float] = {
    "transient":  0.50,   # keep ~50% one-year filers (realistic)
    "short":      0.07,   # 2-3 year filers
    "mid":        0.07,   # 4-5 year filers
    "persistent": 0.36,   # 6-7 year filers (full lifecycle)
}

# ── Fraud pattern mix to preserve ─────────────────────────────────────
# Within fraud persons, approximate source split
# always_fraud vs mixed — kept from explore output
FRAUD_PATTERN_PROPORTIONS: Dict[str, float] = {
    "always_fraud": 0.78,   # chronic evaders (dominant)
    "mixed":        0.22,   # escalators + reformed + sporadic
}

STATE_FILES: Dict[str, str] = {
    "california": r"D:\projiikkkkttttt\data\by_state\california.parquet",
    "illinois":   r"D:\projiikkkkttttt\data\by_state\illinois.parquet",
    "new_york":   r"D:\projiikkkkttttt\data\by_state\new_york.parquet",
    "florida":    r"D:\projiikkkkttttt\data\by_state\florida.parquet",
    "texas":      r"D:\projiikkkkttttt\data\by_state\texas.parquet",
}

OUTPUT_DIR: str = (
    r"D:\tax_evasion_synthetic_data\scripts_and_outputs"
    r"\merged_splits\sampled\output"
)

RANDOM_SEED: int = 42

# ── Tolerance for validation checks ───────────────────────────────────
FRAUD_RATE_TOLERANCE:     float = 0.005
ROW_COUNT_TOLERANCE:      float = 0.02   # 2% of target
LIFECYCLE_TOLERANCE:      float = 0.05   # 5pp drift allowed per bucket