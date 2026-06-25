# 01_core_features.py
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CORE FEATURE ENGINEERING
#
# Key changes in this version:
#   - add_block1_lifestyle: bank_deposit features use log-normalized values
#   - add_block2_tax_burden: withholding signal added
#   - add_block6_financial_ratios: NEW — 13 ratio features using raw cols
#     that are now available because Phase 1 merges raw columns early.
#     All divisions use +1 guard (matching original ddf formulas exactly).
#     Columns are filled with 0 if absent so the block is always safe.
#   - income_band NaN -> "UNKNOWN"
#   - Shannon entropy uses exact masked computation
# ═══════════════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

from config import (
    CLEAN_DIR, ENGINEERED_DIR, YEARS,
    INCOME_SOURCE_COLS, SCH_C_COLS,
    INCOME_BANDS, INCOME_BAND_LABELS, EPS,
)


def compute_vectorized_entropy(df: pd.DataFrame, columns: list) -> np.ndarray:
    """
    Fully vectorized Shannon entropy over a subset of DataFrame columns.
    Exact masked computation — no additive EPS in denominator or log.
    Rows with all-zero values return entropy 0 (minimum diversity).
    """
    present  = [c for c in columns if c in df.columns]
    if not present:
        return np.zeros(len(df), dtype=np.float32)

    matrix   = np.maximum(df[present].fillna(0.0).values, 0.0)
    row_sums = matrix.sum(axis=1)
    nz_mask  = row_sums > 0
    entropy  = np.zeros(len(df), dtype=np.float64)

    if nz_mask.any():
        m_nz      = matrix[nz_mask]
        s_nz      = row_sums[nz_mask, np.newaxis]
        probs     = m_nz / s_nz
        log_probs = np.where(
            probs > 0,
            np.log2(np.where(probs > 0, probs, 1.0)),
            0.0,
        )
        entropy[nz_mask] = -(probs * log_probs).sum(axis=1)

    return entropy.astype(np.float32)


def _safe_col(df: pd.DataFrame, col: str, fill: float = 0.0) -> pd.Series:
    """Return column filled with `fill` if present, else a zero Series."""
    if col in df.columns:
        return df[col].fillna(fill)
    return pd.Series(fill, index=df.index, dtype="float32")


def add_block1_lifestyle(df: pd.DataFrame) -> pd.DataFrame:
    """Block 1: Lifestyle vs income mismatch signals."""

    bdr     = df["bank_deposit_ratio"].fillna(0)
    bdr_log = (
        df["bank_deposit_ratio_log"].fillna(0)
        if "bank_deposit_ratio_log" in df.columns
        else pd.Series(np.log1p(np.maximum(bdr, 0)), index=df.index)
    )

    df["absolute_lifestyle_gap"] = (
        (df["lifestyle_income_ratio"].fillna(0) * df["agi"]) - df["agi"]
    ).clip(-1_000_000, 5_000_000).astype("float32")

    df["bank_deposit_gap"] = (
        (bdr * df["agi"]) - df["agi"]
    ).clip(-1_000_000, 25_000_000).astype("float32")

    df["bank_deposit_gap_ratio"] = (
        df["bank_deposit_gap"] / (df["agi"].abs() + EPS)
    ).clip(-5, 500).astype("float32")

    df["bank_deposit_gap_ratio_log"] = (
        np.log1p(np.maximum(df["bank_deposit_gap_ratio"], 0))
    ).astype("float32")

    df["absolute_lifestyle_spend"] = (
        df["lifestyle_income_ratio"].fillna(0) * df["agi"]
    ).clip(0, 10_000_000).astype("float32")

    df["unexplained_cash_proxy"] = (
        df["bank_deposit_gap"] - df["deduction_taken"].fillna(0)
    ).clip(-1_000_000, 25_000_000).astype("float32")

    df["lifestyle_deposit_combined"] = (
        df["lifestyle_income_ratio"].fillna(0) + bdr_log
    ).clip(0, 30).astype("float32")

    df["income_band"] = (
        pd.cut(
            df["agi"],
            bins=INCOME_BANDS,
            labels=INCOME_BAND_LABELS,
            right=False,
        )
        .cat.add_categories(["UNKNOWN"])
        .fillna("UNKNOWN")
        .astype(str)
    )

    return df


def add_block2_tax_burden(df: pd.DataFrame) -> pd.DataFrame:
    """Block 2: Tax burden signals."""

    df["tax_rate_gap"] = (
        df["expected_tax_rate"] - df["effective_tax_rate"].fillna(0)
    ).clip(-0.5, 0.5).astype("float32")

    df["tax_rate_gap_ratio"] = (
        df["tax_rate_gap"] / (df["expected_tax_rate"].abs() + EPS)
    ).clip(-2, 2).astype("float32")

    df["underpayment_proxy"] = (
        df["tax_rate_gap"] * df["agi"]
    ).clip(-500_000, 1_000_000).astype("float32")

    df["bracket_rate_ratio"] = (
        df["effective_tax_rate"].fillna(0) /
        (df["expected_tax_rate"] + EPS)
    ).clip(0, 2).astype("float32")

    computed_rate = (
        df["total_tax_liability"].fillna(0) / (df["agi"].abs() + EPS)
    ).clip(0, 1)
    df["effective_rate_consistency_error"] = (
        (df["effective_tax_rate"].fillna(0) - computed_rate).abs()
    ).clip(0, 1).astype("float32")

    if "withholding_rate" in df.columns:
        df["low_withholding_indicator"] = (
            1.0 - df["withholding_rate"].fillna(0).clip(0, 1)
        ).astype("float32")
    elif "federal_withheld" in df.columns:
        fw_log = np.log1p(np.maximum(df["federal_withheld"].fillna(0), 0))
        df["low_withholding_indicator"] = (
            1.0 - (fw_log / 12.0).clip(0, 1)
        ).astype("float32")
    else:
        df["low_withholding_indicator"] = np.float32(0.5)

    if "deduction_taken" in df.columns:
        df["progressive_anomaly"] = (
            (df["agi"] > 100_000) &
            (df["effective_tax_rate"].fillna(0) < 0.08) &
            (df["deduction_taken"].fillna(0) < df["agi"] * 0.20)
        ).astype("int8")
    else:
        df["progressive_anomaly"] = (
            (df["agi"] > 100_000) &
            (df["effective_tax_rate"].fillna(0) < 0.08)
        ).astype("int8")

    return df


def add_block3_deductions(df: pd.DataFrame) -> pd.DataFrame:
    """Block 3: Deduction signals."""

    df["deduction_per_1k_agi"] = (
        df["deduction_taken"].fillna(0) /
        (df["agi"].abs() + EPS) * 1000
    ).clip(0, 2000).astype("float32")

    df["net_deduction_rate"] = (
        df["deduction_taken"].fillna(0) /
        (df["agi"].abs() + EPS)
    ).clip(0, 5).astype("float32")

    sch_c_present = [c for c in SCH_C_COLS if c in df.columns]
    df["sch_c_total_expenses"] = (
        df[sch_c_present].fillna(0).sum(axis=1)
    ).astype("float32")

    df["sch_c_to_revenue_ratio"] = (
        df["sch_c_total_expenses"] /
        (df["gross_receipts"].fillna(0) + EPS)
    ).clip(0, 10).astype("float32")

    if "sch_c_meals" in df.columns:
        df["meals_to_revenue_ratio"] = (
            df["sch_c_meals"].fillna(0) /
            (df["gross_receipts"].fillna(0) + EPS)
        ).clip(0, 5).astype("float32")

    if "sch_c_car_truck" in df.columns:
        df["car_to_revenue_ratio"] = (
            df["sch_c_car_truck"].fillna(0) /
            (df["gross_receipts"].fillna(0) + EPS)
        ).clip(0, 5).astype("float32")

    if "sch_c_home_office" in df.columns:
        df["home_office_w2_flag"] = (
            (df["sch_c_home_office"].fillna(0) > 0) &
            (df["w2_wages"].fillna(0) > 0) &
            (df["net_se_income"].fillna(0) == 0)
        ).astype("int8")

    df["phantom_business_flag"] = (
        (df["sch_c_total_expenses"] > 500) &
        (df["gross_receipts"].fillna(0) == 0)
    ).astype("int8")

    df["se_perpetual_loss"] = (
        df["net_se_income"].fillna(0) < -1000
    ).astype("int8")

    if len(sch_c_present) > 1:
        df["expense_concentration"] = (
            df[sch_c_present].fillna(0).max(axis=1) /
            (df["sch_c_total_expenses"] + EPS)
        ).clip(0, 1).astype("float32")
    else:
        df["expense_concentration"] = np.float32(1.0)

    df["expense_entropy"] = compute_vectorized_entropy(df, sch_c_present)

    return df


def add_block4_income_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Block 4: Income source signals."""

    agi_abs = df["agi"].abs() + EPS

    df["w2_share_of_agi"]         = (df["w2_wages"].fillna(0)         / agi_abs).clip(0,    1).astype("float32")
    df["se_share_of_agi"]         = (df["net_se_income"].fillna(0)    / agi_abs).clip(-0.5, 1).astype("float32")
    df["gig_share_of_agi"]        = (df["gig_net"].fillna(0)          / agi_abs).clip(0,    1).astype("float32")
    df["rental_share_of_agi"]     = (df["rental_net"].fillna(0)       / agi_abs).clip(-0.5, 1).astype("float32")
    df["investment_share_of_agi"] = (
        (df["dividends"].fillna(0) +
         df["capital_gains_lt"].fillna(0) +
         df["interest_income"].fillna(0)) / agi_abs
    ).clip(0, 1).astype("float32")

    income_cols_present = [c for c in INCOME_SOURCE_COLS if c in df.columns]
    df["income_stream_count"] = (
        df[income_cols_present].fillna(0).gt(0).sum(axis=1)
    ).astype("int8")

    df["income_entropy"] = compute_vectorized_entropy(df, income_cols_present)

    df["income_concentration"] = (
        df[income_cols_present].fillna(0).max(axis=1) / agi_abs
    ).clip(0, 1).astype("float32")

    df["has_cash_business"] = (
        df["gross_receipts"].fillna(0) > 0
    ).astype("int8")

    df["cash_business_dominance"] = (
        df["gross_receipts"].fillna(0) / agi_abs
    ).clip(0, 5).astype("float32")

    df["se_loss_offsetting_w2"] = (
        (df["net_se_income"].fillna(0) < -1000) &
        (df["w2_wages"].fillna(0) > 0)
    ).astype("int8")

    df["income_type_mismatch"] = (
        (df["taxpayer_type"] == "pure_w2") &
        (
            (df["gig_share_of_agi"] > 0.10) |
            (df["se_share_of_agi"]  > 0.10)
        )
    ).astype("int8")

    return df


def add_block5_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Block 5: Internal consistency signals."""

    df["impossible_w2_exceeds_agi"] = (
        df["w2_wages"].fillna(0) > df["agi"] + 100
    ).astype("int8")

    df["deduction_exceeds_agi"] = (
        df["deduction_taken"].fillna(0) > df["agi"] + 100
    ).astype("int8")

    df["impossible_negative_tax_high_income"] = (
        (df["total_tax_liability"].fillna(0) < 0) &
        (df["agi"] > 100_000)
    ).astype("int8")

    df["impossible_rate_above_one"] = (
        df["effective_tax_rate"].fillna(0) > 1.0
    ).astype("int8")

    if "taxable_income" in df.columns:
        expected_taxable = df["agi"] - df["deduction_taken"].fillna(0)
        df["taxable_income_error"] = (
            (df["taxable_income"] - expected_taxable).abs()
        ).clip(0, 1_000_000).astype("float32")
        df["taxable_income_error_ratio"] = (
            df["taxable_income_error"] / (df["agi"].abs() + EPS)
        ).clip(0, 5).astype("float32")

        reconstructed_tax = (
            df["taxable_income"].fillna(0) * df["effective_tax_rate"].fillna(0)
        )
        df["tax_liability_error"] = (
            (df["total_tax_liability"].fillna(0) - reconstructed_tax).abs()
        ).clip(0, 500_000).astype("float32")

    df["expense_without_revenue_amount"] = (
        df["sch_c_total_expenses"] *
        (df["gross_receipts"].fillna(0) == 0).astype(int)
    ).clip(0, 1_000_000).astype("float32")

    df["business_profit_margin"] = np.where(
        df["gross_receipts"].fillna(0) > 0,
        (
            (df["gross_receipts"].fillna(0) - df["sch_c_total_expenses"]) /
            (df["gross_receipts"].fillna(0) + EPS)
        ).clip(-2, 1),
        np.nan,
    ).astype("float32")

    df["negative_margin_flag"] = (
        df["business_profit_margin"].fillna(0) < -0.1
    ).astype("int8")

    return df


def add_block6_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Block 6: Financial ratio features using raw columns merged in Phase 1.

    All raw columns are accessed via _safe_col() which returns a zero
    Series when the column is absent — safe for any dataset variant.

    Division guard uses +1 (matching original formula exactly) rather
    than EPS so that near-zero denominators produce bounded ratios
    consistent with the original specification.

    New features
    ────────────
    credit_to_income            : tax credits relative to AGI
    debt_to_income              : mortgage interest relative to AGI
    mortgage_leverage_ratio     : alias of debt_to_income (different
                                  downstream meaning in audit context)
    credit_dependency_index     : credits vs pre-credit tax liability
    deduction_aggression_score  : deductions vs ALL earned income streams
    cogs_to_receipts_ratio      : COGS vs gross business receipts
    sch_c_wages_to_revenue      : business payroll vs revenue
    gig_expense_ratio           : gig expenses vs gig income
    crypto_loss_shield_ratio    : crypto cost basis vs proceeds
                                  (>1 = net loss position used as shield)
    utility_cost_per_rental_unit: utilities per rental unit
    passive_income_density      : dividends + interest vs AGI
    new_business_interaction    : Sch C presence × first-year filing flag
    w2_withholding_accuracy     : total federal withheld vs W2 wages
    """
    agi = df["agi"].fillna(0)

    # ── credit_to_income ──────────────────────────────────────────────────────
    total_credits = _safe_col(df, "total_credits")
    df["credit_to_income"] = (
        total_credits / (agi + 1)
    ).clip(-5, 5).astype("float32")

    # ── debt_to_income ────────────────────────────────────────────────────────
    mortgage_int = _safe_col(df, "itemized_mortgage_int")
    df["debt_to_income"] = (
        mortgage_int / (agi + 1)
    ).clip(0, 5).astype("float32")

    # ── mortgage_leverage_ratio ───────────────────────────────────────────────
    # Kept as separate column per original spec (different audit interpretation)
    df["mortgage_leverage_ratio"] = (
        mortgage_int / (agi + 1)
    ).clip(0, 5).astype("float32")

    # ── credit_dependency_index ───────────────────────────────────────────────
    tax_before_credits = _safe_col(df, "tax_before_credits")
    df["credit_dependency_index"] = (
        total_credits / (tax_before_credits + 1)
    ).clip(0, 5).astype("float32")

    # ── deduction_aggression_score ────────────────────────────────────────────
    # Sum of ALL earned income streams as denominator — aggressive deductions
    # relative to ALL income sources (not just AGI) is a stronger fraud signal
    deduction_taken = _safe_col(df, "deduction_taken")
    w2_wages        = _safe_col(df, "w2_wages")
    gross_receipts  = _safe_col(df, "gross_receipts")
    gig_income      = _safe_col(df, "gig_income")
    rental_gross    = _safe_col(df, "rental_gross")
    crypto_proceeds = _safe_col(df, "crypto_proceeds")

    total_earned = (
        w2_wages + gross_receipts + gig_income +
        rental_gross + crypto_proceeds + 1
    )
    df["deduction_aggression_score"] = (
        deduction_taken / total_earned
    ).clip(0, 10).astype("float32")

    # ── cogs_to_receipts_ratio ────────────────────────────────────────────────
    cogs = _safe_col(df, "cogs")
    df["cogs_to_receipts_ratio"] = (
        cogs / (gross_receipts + 1)
    ).clip(0, 5).astype("float32")

    # ── sch_c_wages_to_revenue ────────────────────────────────────────────────
    sch_c_wages = _safe_col(df, "sch_c_wages")
    df["sch_c_wages_to_revenue"] = (
        sch_c_wages / (gross_receipts + 1)
    ).clip(0, 5).astype("float32")

    # ── gig_expense_ratio ─────────────────────────────────────────────────────
    gig_expenses = _safe_col(df, "gig_expenses")
    df["gig_expense_ratio"] = (
        gig_expenses / (gig_income + 1)
    ).clip(0, 10).astype("float32")

    # ── crypto_loss_shield_ratio ──────────────────────────────────────────────
    # Values > 1 indicate the taxpayer holds a net loss position;
    # extreme values (e.g., 10×) suggest wash-sale or phantom-basis schemes
    crypto_cost_basis = _safe_col(df, "crypto_cost_basis")
    df["crypto_loss_shield_ratio"] = (
        crypto_cost_basis / (crypto_proceeds + 1)
    ).clip(0, 20).astype("float32")

    # ── utility_cost_per_rental_unit ──────────────────────────────────────────
    utility_cost   = _safe_col(df, "utility_cost_estimated")
    n_rental_units = _safe_col(df, "n_rental_units", fill=0.0)
    df["utility_cost_per_rental_unit"] = (
        utility_cost / (n_rental_units + 1)
    ).clip(0, 50_000).astype("float32")

    # ── passive_income_density ────────────────────────────────────────────────
    dividends       = _safe_col(df, "dividends")
    interest_income = _safe_col(df, "interest_income")
    df["passive_income_density"] = (
        (dividends + interest_income) / (agi + 1)
    ).clip(0, 5).astype("float32")

    # ── new_business_interaction ──────────────────────────────────────────────
    # Multiplicative interaction: only fires when BOTH are true.
    # First-year filers with Sch C income have elevated audit risk.
    has_schedule_c   = _safe_col(df, "has_schedule_c")
    first_year_filing = _safe_col(df, "first_year_filing")
    df["new_business_interaction"] = (
        has_schedule_c * first_year_filing
    ).clip(0, 1).astype("float32")

    # ── w2_withholding_accuracy ───────────────────────────────────────────────
    # federal_withheld_total (Form W-2 box 2 sum) vs W2 wages.
    # Ratio >> expected marginal rate suggests over-withholding (rare fraud).
    # Ratio << expected rate (or 0) suggests under-withholding / evasion.
    federal_withheld_total = _safe_col(df, "federal_withheld_total")
    df["w2_withholding_accuracy"] = (
        federal_withheld_total / (w2_wages + 1)
    ).clip(0, 2).astype("float32")

    return df


def add_temporal_context(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal and demographic context features."""

    df["years_since_base"] = (df["tax_year"] - 2019).astype("int8")
    df["is_covid_year"]    = df["tax_year"].isin([2020, 2021]).astype("int8")
    df["is_recent_year"]   = df["tax_year"].isin([2024, 2025]).astype("int8")

    df["age_group"] = pd.cut(
        df["age"].fillna(45),
        bins=[0, 25, 35, 50, 65, 200],
        labels=["under_25", "25_35", "35_50", "50_65", "over_65"],
    ).astype(str)

    df["is_near_retirement"]     = df["age"].fillna(0).between(60, 70).astype("int8")
    df["is_young_high_earner"]   = ((df["age"].fillna(0) < 30) & (df["agi"] > 150_000)).astype("int8")
    df["retired_high_se_income"] = ((df["age"].fillna(0) > 65) & (df["net_se_income"].fillna(0) > 50_000)).astype("int8")

    return df


def run():
    print("=" * 65)
    print("PHASE 2 -- CORE FEATURE ENGINEERING (Blocks 1-6 + Temporal)")
    print("=" * 65)

    for year in YEARS:
        p_in  = CLEAN_DIR      / f"year_{year}_clean.parquet"
        p_out = ENGINEERED_DIR / f"year_{year}_engineered.parquet"

        if not p_in.exists():
            print(f"  year_{year}: clean file MISSING -- skipped")
            continue

        print(f"\n  year_{year}: loading...", end="")
        df       = pd.read_parquet(p_in)
        n_before = df.shape[1]
        print(f" {len(df):,} rows x {n_before} cols")

        df = add_block1_lifestyle(df);          print(f"    Block 1 [OK] lifestyle")
        df = add_block2_tax_burden(df);         print(f"    Block 2 [OK] tax burden")
        df = add_block3_deductions(df);         print(f"    Block 3 [OK] deductions")
        df = add_block4_income_sources(df);     print(f"    Block 4 [OK] income sources")
        df = add_block5_consistency(df);        print(f"    Block 5 [OK] consistency")
        df = add_block6_financial_ratios(df);   print(f"    Block 6 [OK] financial ratios")
        df = add_temporal_context(df);          print(f"    Block 7 [OK] temporal context")

        new_b6 = [
            "credit_to_income", "debt_to_income", "mortgage_leverage_ratio",
            "credit_dependency_index", "deduction_aggression_score",
            "cogs_to_receipts_ratio", "sch_c_wages_to_revenue",
            "gig_expense_ratio", "crypto_loss_shield_ratio",
            "utility_cost_per_rental_unit", "passive_income_density",
            "new_business_interaction", "w2_withholding_accuracy",
        ]
        present_b6 = [c for c in new_b6 if c in df.columns]
        print(f"    Block 6 features present : {len(present_b6)}/13")

        print(f"    New features : {df.shape[1] - n_before}")
        print(f"    Total cols   : {df.shape[1]}")

        df.to_parquet(p_out, index=False)
        print(f"    Saved -> {p_out.name}")
        del df; gc.collect()

    print(f"\n{'=' * 65}")
    print(f"PHASE 2 COMPLETE -- Output: {ENGINEERED_DIR}")


if __name__ == "__main__":
    run()