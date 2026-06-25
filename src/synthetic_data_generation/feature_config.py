# feature_config.py
"""
Defines which columns are used for what purpose.
Import this in your training, GAN, and FL scripts.
Never hardcode column lists anywhere else.
"""

# ── Columns never used in training ───────────────────────────────────────────
# Either reveal the answer or are not observable in real life

DROP_ALWAYS = [
    # Reveals fraud mechanism directly
    "total_income_pre_fraud",
    "fraud_persona",
    "evasion_amount",
    "evasion_rate",
    "tax_gap_amount",
    "true_tax_liability",
    # Fraud amount cols — the hidden amounts are not observable
    "unreported_cash_amount",
    "amt_1099_not_reported",
    "gig_omitted_amount",
    "rental_hidden_amount",
    "farm_hidden_amount",
    "crypto_unreported_amount",
    "offshore_hidden_amount",
    "fictitious_ded_amount",
    "expense_rechar_amount",
    "inflated_cogs_amount",
    "low_salary_scorp_amount",
    "payroll_underrpt_amount",
    "worker_misclass_amount",
    "revenue_suppression_amount",
    "cash_skimming_amount",
    "shell_shifting_amount",
    "capital_gains_omit_amount",
]

# ── ID columns — used for grouping/tracking, not features ────────────────────
ID_COLS = [
    "person_id",
    "employer_id",
]

# ── Target columns ────────────────────────────────────────────────────────────
TARGET_PRIMARY   = "fraud_label"       # binary 0/1
TARGET_TYPE      = "fraud_type"        # multiclass fraud type
TARGET_CATEGORY  = "fraud_category"    # broader category

# ── Categorical features ──────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "filing_status",
    "taxpayer_type",
    "primary_occupation",
    "sex",
    "education",
]

# ── Binary flag features ──────────────────────────────────────────────────────
BINARY_COLS = [
    "has_schedule_c",
    "has_gig",
    "has_rental",
    "has_investments",
    "has_crypto",
    "has_foreign_account",
    "received_ppp",
    "uses_itemized",
    "first_year_filing",
    "fbar_required",
]

# ── Numeric features ──────────────────────────────────────────────────────────
NUMERIC_COLS = [
    # Temporal / geographic
    "tax_year",
    "zone",
    "age",
    "entry_cohort",

    # W2 employment
    "w2_wages",
    "federal_withheld",
    "fica_withheld",
    "medicare_withheld",
    "federal_withheld_total",

    # Schedule C
    "gross_receipts",
    "cogs",
    "gross_profit",
    "total_expenses",
    "net_se_income",
    "sch_c_advertising",
    "sch_c_car_truck",
    "sch_c_depreciation",
    "sch_c_insurance",
    "sch_c_meals",
    "sch_c_office_expense",
    "sch_c_rent",
    "sch_c_repairs",
    "sch_c_supplies",
    "sch_c_utilities",
    "sch_c_wages",
    "sch_c_home_office",

    # Gig
    "gig_income",
    "gig_expenses",
    "gig_net",

    # Rental
    "rental_gross",
    "rental_expenses",
    "rental_depreciation",
    "rental_net",
    "n_rental_units",

    # Investments
    "dividends",
    "capital_gains_lt",
    "capital_gains_st",
    "interest_income",

    # Crypto
    "crypto_proceeds",
    "crypto_cost_basis",
    "crypto_net_gain",

    # Foreign
    "foreign_income",
    "foreign_account_balance",

    # Business owner
    "k1_income",
    "owner_salary",

    # Retirement / other
    "social_security_income",
    "pension_income",
    "ira_distributions",
    "unemployment_comp",

    # Deductions
    "standard_deduction",
    "itemized_total",
    "itemized_mortgage_int",
    "itemized_salt",
    "itemized_charitable",
    "itemized_medical",
    "deduction_taken",
    "qbi_deduction",
    "se_tax_amount",
    "se_tax_deduction",

    # Tax calculations
    "agi",
    "taxable_income",
    "tax_before_credits",
    "eitc_credit",
    "child_tax_credit",
    "total_credits",
    "total_tax_liability",
    "effective_tax_rate",
    "refund_amount",
    "balance_due",

    # Utilities / lifestyle
    "electricity_kwh",
    "gas_therms",
    "water_gallons",
    "utility_cost_estimated",
    "utility_income_ratio",

    # Key fraud detection signals
    "lifestyle_income_ratio",
    "bank_deposit_ratio",
    "deduction_income_ratio",
    "effective_rate_vs_zone",
    "irs_risk_score",

    # PPP
    "ppp_loan_amount",
]

# ── All training features combined ────────────────────────────────────────────
FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS

# ── Columns to keep in parquet but never use in training ─────────────────────
AUDIT_ONLY_COLS = DROP_ALWAYS + ID_COLS + [
    TARGET_PRIMARY,
    TARGET_TYPE,
    TARGET_CATEGORY,
]


def get_feature_cols(df_columns: list) -> list:
    """
    Return feature cols that actually exist in the dataframe.
    Safe to call on any split or GAN output.
    """
    return [c for c in FEATURE_COLS if c in df_columns]


def get_training_data(df):
    """
    Return X (features) and y (target) ready for training.
    Drops all audit-only and unobservable cols automatically.
    """
    feature_cols = get_feature_cols(df.columns.tolist())
    X = df[feature_cols].copy()
    y = df[TARGET_PRIMARY].copy()
    return X, y


def get_col_types(df_columns: list) -> dict:
    """
    Return dict of col_name -> type for existing columns only.
    Useful for encoders and scalers that need to know col types.
    """
    existing_num = [c for c in NUMERIC_COLS  if c in df_columns]
    existing_cat = [c for c in CATEGORICAL_COLS if c in df_columns]
    existing_bin = [c for c in BINARY_COLS   if c in df_columns]
    return {
        "numeric":     existing_num,
        "categorical": existing_cat,
        "binary":      existing_bin,
    }