# utils.py
"""
Shared utilities: logging, parquet I/O, fraud-rate enforcement,
null-policy application, and path helpers.
"""

from config import DIST_PKL, LOGS_BASE, RANDOM_SEED, VOLUME_NAMES, DIST_BASE

import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import DIST_PKL, LOGS_BASE, RANDOM_SEED


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a logger that writes to stdout and optionally to a UTF-8 file.

    Safe on Windows cp1252 terminals and Modal Linux containers alike.
    """
    LOGS_BASE.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    if not log.handlers:
        # ── stdout handler ────────────────────────────────────────────────
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)

        # ── optional file handler ─────────────────────────────────────────
        if log_file:
            fh = logging.FileHandler(
                LOGS_BASE / log_file,
                encoding="utf-8",
            )
            fh.setFormatter(fmt)
            log.addHandler(fh)

    return log


# ── Path helpers ──────────────────────────────────────────────────────────────

def make_dirs(*paths: Path) -> None:
    """Create every supplied directory (and parents) if it doesn't exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ── Distributions I/O ─────────────────────────────────────────────────────────

def load_distributions() -> dict:
    """
    Load master_distributions.pkl from DIST_PKL (config-defined path).

    On Modal this resolves to /distributions/master_distributions.pkl
    which must be present in the 'tax-distributions' volume.
    Raises FileNotFoundError with a clear message if the file is missing.
    """
    if not DIST_PKL.exists():
        raise FileNotFoundError(
            f"Distributions file not found: {DIST_PKL}\n"
            "On Modal: ensure 'tax-distributions' volume is mounted at "
            "/distributions and master_distributions.pkl has been uploaded."
        )
    with open(DIST_PKL, "rb") as f:
        return pickle.load(f)


# ── Parquet I/O ───────────────────────────────────────────────────────────────

def write_parquet(
    df: pd.DataFrame,
    path: Path,
    compression: str = "snappy",
) -> None:
    """Write df to path as a Parquet file; create parent dirs automatically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression=compression,
        write_statistics=True,
        row_group_size=50_000,
    )
    mb = path.stat().st_size / 1e6
    print(f"  wrote {len(df):>10,} rows -> {path}  ({mb:.1f} MB)")


def read_parquet(path: Path, **kwargs) -> pd.DataFrame:
    """Thin wrapper around pd.read_parquet for consistent call sites."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path, **kwargs)


# ── Null-policy enforcement ───────────────────────────────────────────────────

# Columns that are structurally null unless the person meets the condition.
# Use np.nan (not pd.NA) so float64 columns stay float64 without dtype
# promotion to object or pandas nullable types.
STRUCTURAL_NULL_MAP = {
    "_se_cols": {
        # Present only for taxpayer types that have self-employment income
        "null_when_type_not_in": [
            "pure_se",
            "w2_with_side_biz",
            "business_owner",
            "multi_biz_owner",
            "gig_only",
        ],
        "columns": [
            "gross_receipts", "cogs", "gross_profit",
            "total_expenses", "net_se_income", "has_schedule_c",
            "sch_c_advertising", "sch_c_car_truck", "sch_c_depreciation",
            "sch_c_insurance", "sch_c_meals", "sch_c_office_expense",
            "sch_c_rent", "sch_c_repairs", "sch_c_supplies",
            "sch_c_utilities", "sch_c_wages", "sch_c_home_office",
            "sch_c_other", "se_tax", "se_tax_deduction",
        ],
    },
    "_rental_cols": {
        # Present only when has_rental == 1
        "null_when_flag": ("has_rental", 0),
        "columns": [
            "rental_gross", "rental_expenses",
            "rental_depreciation", "rental_net", "n_rental_units",
        ],
    },
    "_crypto_cols": {
        # Present only when has_crypto == 1
        "null_when_flag": ("has_crypto", 0),
        "columns": [
            "crypto_proceeds", "crypto_cost_basis", "crypto_net_gain",
        ],
    },
    "_foreign_cols": {
        # Present only when has_foreign == 1
        "null_when_flag": ("has_foreign", 0),
        "columns": [
            "foreign_income", "foreign_account_balance", "fbar_required",
        ],
    },
    "_ss_cols": {
        # Social Security income only for persons aged >= 62
        "null_when_age_lt": 62,
        "columns": ["social_security_income"],
    },
    "_fraud_amt_cols": {
        # Evasion amounts only meaningful when fraud_label == 1
        "null_when_flag": ("fraud_label", 0),
        "columns": [
            "evasion_amount",
            "evasion_rate",
        ],
    },
}

# All cols that are known strings — everything else numeric gets coerced
KNOWN_STRING_COLS = {
    "person_id", "sex", "education", "filing_status",
    "taxpayer_type", "primary_occupation", "employer_id",
    "fraud_type", "fraud_category", "fraud_persona",
}

def apply_null_policy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set structurally-absent values to np.nan based on taxpayer type / flags.
    Also enforces dtype consistency on all numeric columns.
    """
    df = df.copy()

    # ── SE columns ────────────────────────────────────────────────────────────
    if "taxpayer_type" in df.columns:
        rule    = STRUCTURAL_NULL_MAP["_se_cols"]
        se_mask = ~df["taxpayer_type"].isin(rule["null_when_type_not_in"])
        for col in rule["columns"]:
            if col in df.columns:
                df.loc[se_mask, col] = np.nan

    # ── Flag-based columns ────────────────────────────────────────────────────
    for key in (
        "_rental_cols", "_crypto_cols", "_foreign_cols", "_fraud_amt_cols"
    ):
        rule               = STRUCTURAL_NULL_MAP[key]
        flag_col, flag_val = rule["null_when_flag"]
        if flag_col not in df.columns:
            continue
        null_mask = df[flag_col] == flag_val
        for col in rule["columns"]:
            if col in df.columns:
                df.loc[null_mask, col] = np.nan

    # ── Age-gated columns ─────────────────────────────────────────────────────
    if "age" in df.columns:
        rule     = STRUCTURAL_NULL_MAP["_ss_cols"]
        age_mask = df["age"] < rule["null_when_age_lt"]
        for col in rule["columns"]:
            if col in df.columns:
                df.loc[age_mask, col] = np.nan

    # ── W2 zero-wage protection ───────────────────────────────────────────────
    if "w2_wages" in df.columns and "taxpayer_type" in df.columns:
        zero_mask = (
            df["taxpayer_type"].isin(["pure_w2", "w2_with_side_biz"])
            & (df["w2_wages"].fillna(0) == 0)
        )
        if zero_mask.any():
            for col in [
                "w2_wages", "federal_withheld", "fica_withheld",
                "medicare_withheld", "federal_withheld_total",
            ]:
                if col in df.columns:
                    df.loc[zero_mask, col] = np.nan

    # ── Coerce all non-string object cols to float64 ──────────────────────────
    # Handles ALL fraud amount cols and any other numeric col that
    # arrived as object due to mixed None/float in row-dict construction
    for col in df.select_dtypes("object").columns:
        if col not in KNOWN_STRING_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Force int64 on binary flag columns ────────────────────────────────────
    INT_FLAGS = [
        "has_schedule_c", "has_gig", "has_rental", "has_investments",
        "has_crypto", "has_foreign_account", "received_ppp",
        "uses_itemized", "first_year_filing", "fraud_label",
    ]
    for col in INT_FLAGS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("int64")

    return df

# ── Fraud-rate enforcement ────────────────────────────────────────────────────

def enforce_fraud_rate(
    df:           pd.DataFrame,
    target_rate:  float,
    tolerance:    float = 0.015,
    label_col:    str   = "fraud_label",
    random_state: int   = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Resample df so fraud_label fraction == target_rate +/- tolerance.
    """
    evaders   = df[df[label_col] == 1]
    compliant = df[df[label_col] == 0]
    n_total   = len(df)
    actual    = len(evaders) / max(n_total, 1)

    if abs(actual - target_rate) <= tolerance:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_evaders_target   = int(round(n_total * target_rate))
    n_compliant_target = n_total - n_evaders_target

    if len(evaders) >= n_evaders_target:
        evaders_out = evaders.sample(
            n=n_evaders_target, replace=False, random_state=random_state
        )
    else:
        evaders_out = evaders.sample(
            n=n_evaders_target, replace=True, random_state=random_state
        )

    if len(compliant) >= n_compliant_target:
        compliant_out = compliant.sample(
            n=n_compliant_target, replace=False, random_state=random_state
        )
    else:
        compliant_out = compliant.sample(
            n=n_compliant_target, replace=True, random_state=random_state
        )

    result = (
        pd.concat([evaders_out, compliant_out])
          .sample(frac=1, random_state=random_state)
          .reset_index(drop=True)
    )

    final_rate = result[label_col].mean()
    print(f"  fraud rate enforced: {final_rate:.4f} (target {target_rate:.4f})")
    return result


def compute_tax_liability(
    taxable_income: float,
    year:           int,
    filing_status:  str,
) -> float:
    """
    Compute federal income tax using progressive brackets from config.

    Filing status handling:
      married_joint    : all bracket thresholds doubled
      married_separate : single brackets (same thresholds)
      head_of_household: single brackets (simplified)
      qualifying_widow : same as married_joint
      single           : single brackets

    Returns tax rounded to 2 decimal places.
    """
    from config import TAX_BRACKETS

    if taxable_income <= 0:
        return 0.0

    brackets = TAX_BRACKETS.get(year, TAX_BRACKETS[2023])

    # MFJ and qualifying widow get doubled thresholds
    if filing_status in ("married_joint", "qualifying_widow"):
        brackets = [
            (rate, bound * 2 if bound != float("inf") else float("inf"))
            for rate, bound in brackets
        ]

    tax      = 0.0
    prev_top = 0.0

    for rate, top in brackets:
        slice_top = min(taxable_income, top)
        if slice_top <= prev_top:
            break
        tax     += (slice_top - prev_top) * rate
        prev_top = top
        if taxable_income <= top:
            break

    return round(tax, 2)

# ── Tax computation ──────────────────────────────────────────────────────────

def compute_tax_liability(
    taxable_income: float,
    year:           int,
    filing_status:  str,
) -> float:
    """
    Compute federal income tax using progressive brackets from config.

    Filing status handling:
      - married_joint, qualifying_widow: bracket thresholds doubled  
      - married_separate, head_of_household, single: single brackets

    Returns tax rounded to 2 decimals.
    """
    from config import TAX_BRACKETS

    if taxable_income <= 0:
        return 0.0

    brackets = TAX_BRACKETS.get(year, TAX_BRACKETS[2023])

    # Double thresholds for MFJ and qualifying widow
    if filing_status in ("married_joint", "qualifying_widow"):
        brackets = [
            (rate, bound * 2 if bound != float("inf") else float("inf"))
            for rate, bound in brackets
        ]

    tax = 0.0
    prev_top = 0.0

    for rate, top in brackets:
        slice_top = min(taxable_income, top)
        if slice_top <= prev_top:
            break
        tax += (slice_top - prev_top) * rate
        prev_top = top
        if taxable_income <= top:
            break

    return round(tax, 2)

def build_volumes(volume_map: dict) -> dict:
    """
    Convert dict of mount_point -> volume_name (str) into
    mount_point -> modal.Volume objects.
    """
    import modal
    
    result = {}
    for mount_point, volume_name in volume_map.items():
        mount = str(mount_point)
        vol = modal.Volume.from_name(volume_name, create_if_missing=True)
        result[mount] = vol
    return result