# 08_ctab_gan_runner.py
"""
CTAB-GAN+ (via CTGAN) training for individual tax records.
Three models: w2, se, itemizers.

Run:
  modal run 08_ctab_gan_runner.py::train_w2
  modal run 08_ctab_gan_runner.py::train_se
  modal run 08_ctab_gan_runner.py::train_itemizers
"""

import modal
import pandas as pd
import numpy as np
from pathlib import Path

# ── Image ─────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        # Install torch first alone — avoids conflicts
        "torch==2.2.0",
    ])
    .pip_install([
        # Then data stack
        "pandas==2.2.0",
        "numpy==1.26.4",
        "pyarrow==15.0.0",
        "scikit-learn==1.4.0",
    ])
    .pip_install([
        # Then GAN stack — let it resolve its own deps
        "ctgan",
        "sdv",
    ])
    .add_local_file("config.py", "/root/config.py")
    .add_local_file("utils.py",  "/root/utils.py")
)

# ── App & volumes ─────────────────────────────────────────────────────────────
from config import VOLUME_NAMES

app        = modal.App("taxxx-pipeline-08-ctab-gan")
final_vol  = modal.Volume.from_name(VOLUME_NAMES["final"],   create_if_missing=True)
panels_vol = modal.Volume.from_name(VOLUME_NAMES["panels"],  create_if_missing=True)
logs_vol   = modal.Volume.from_name(VOLUME_NAMES["logs"],    create_if_missing=True)
gan_vol    = modal.Volume.from_name(VOLUME_NAMES["gan_out"], create_if_missing=True) 
VOLUMES = {
    "/final_dataset": final_vol,
    "/seed_panels":   panels_vol,
    "/logs":          logs_vol,
    "/gan_output":    gan_vol,
}

# ── Column declarations ───────────────────────────────────────────────────────
# These must match column names produced by 05_generate_panels.py exactly.

DISCRETE_W2: list[str] = [
    "zone", "filing_status", "sex", "education",
    "taxpayer_type", "primary_occupation",
    "fraud_label", "fraud_type", "fraud_category",
    "has_rental",       # bool flag: 0/1
    "has_investments",  # bool flag: 0/1
    "has_gig",          # bool flag: 0/1
    "has_crypto",       # bool flag: 0/1
    "has_foreign_account",  # bool flag: 0/1
    "uses_itemized",    # bool flag: 0/1
    "has_schedule_c",   # bool flag: 0/1
    "fbar_required",    # 0/1/None → filled to 0 before training
]

DISCRETE_SE: list[str] = DISCRETE_W2 + [
    # "entity_type" is a business attribute, NOT on person-year rows — removed
]

DISCRETE_ITEM: list[str] = [
    "zone", "filing_status", "sex", "education","primary_occupation",
    "taxpayer_type", "fraud_label", "fraud_type",
    "fraud_category", "uses_itemized",
]

# Columns to drop before GAN training
# "tax_year" (not "year") matches script 05 output
# evasion_amount / evasion_rate are None for non-evaders (null policy correct)
# total_income_pre_fraud leaks true income before fraud adjustment
DROP_COLS: list[str] = [
    "person_id",
    "tax_year",
    "entry_cohort",
    "first_year_filing",
    "employer_id",
]


def _prep_df(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    """
    Drop non-trainable columns, fill NAs with safe defaults.

    Categorical NA fill:
      - fraud_type / fraud_category → "none"  (structurally correct default)
      - All other categoricals → column mode (preserves distribution)
      - Numeric → 0.0
    """
    df   = df.copy()
    drop = [c for c in drop_cols if c in df.columns]
    df   = df.drop(columns=drop)

    # Per-column NA fill
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        dtype = df[col].dtype
        if dtype in ("float64", "float32", np.float64, np.float32) or \
                pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].fillna(0.0)
        elif pd.api.types.is_integer_dtype(dtype):
            df[col] = df[col].fillna(0)
        elif col in ("fraud_type", "fraud_category"):
            # Structural default: missing means no fraud
            df[col] = df[col].fillna("none")
        else:
            # Use mode for all other categoricals to preserve distribution
            mode_val = df[col].mode()
            df[col]  = df[col].fillna(
                mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
            )

    return df


def _apply_post_constraints(
    df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Hard post-generation constraints to correct GAN artifacts.
    Column names must match script 05 output exactly.
    """
    df = df.copy()

    # ── Universal constraints ─────────────────────────────────────────────────
    if "agi" in df.columns:
        df["agi"] = df["agi"].clip(lower=0.0)
    if "taxable_income" in df.columns:
        df["taxable_income"] = df["taxable_income"].clip(lower=0.0)
        if "agi" in df.columns:
            # taxable income cannot exceed AGI
            df["taxable_income"] = df[["taxable_income", "agi"]].min(axis=1)
    if "effective_tax_rate" in df.columns:
        df["effective_tax_rate"] = df["effective_tax_rate"].clip(0.0, 0.37)
    if "age" in df.columns:
        df["age"] = df["age"].clip(18, 85).round().astype(int)

    # ── W-2 specific ──────────────────────────────────────────────────────────
    if model_name == "w2":
        if "w2_wages" in df.columns:
            df["w2_wages"] = df["w2_wages"].clip(lower=0.0)
        # Social Security only for age >= 62 (matches null policy in utils)
        if "social_security_income" in df.columns and "age" in df.columns:
            df.loc[df["age"] < 62, "social_security_income"] = np.nan

    # ── SE specific ───────────────────────────────────────────────────────────
    if model_name == "se":
        for col in ("gross_receipts", "cogs", "total_expenses"):
            if col in df.columns:
                df[col] = df[col].clip(lower=0.0)
        # COGS cannot exceed gross receipts
        if "gross_receipts" in df.columns and "cogs" in df.columns:
            df["cogs"] = df[["cogs", "gross_receipts"]].min(axis=1)
        # Recompute net SE income from components for internal consistency
        if all(c in df.columns for c in
               ("gross_receipts", "cogs", "total_expenses", "net_se_income")):
            df["net_se_income"] = (
                df["gross_receipts"]
                - df["cogs"].fillna(0.0)
                - df["total_expenses"].fillna(0.0)
            ).clip(lower=-200_000.0)

    # ── Itemizer specific ─────────────────────────────────────────────────────
    if model_name == "itemizers":
        # SALT cap: $10,000 (script 05 column is "itemized_salt")
        if "itemized_salt" in df.columns:
            df["itemized_salt"] = df["itemized_salt"].clip(0.0, 10_000.0)
        # By definition all rows in itemizer model itemize
        if "uses_itemized" in df.columns:
            df["uses_itemized"] = 1

    # ── Fraud consistency ─────────────────────────────────────────────────────
    if "fraud_label" in df.columns:
        df["fraud_label"] = df["fraud_label"].clip(0, 1).round().astype(int)
        clean_mask = df["fraud_label"] == 0
        if "fraud_type" in df.columns:
            df.loc[clean_mask, "fraud_type"]     = "none"
        if "fraud_category" in df.columns:
            df.loc[clean_mask, "fraud_category"] = "none"

    # ── total_tax_liability (script 05 name) ──────────────────────────────────
    if "total_tax_liability" in df.columns:
        df["total_tax_liability"] = df["total_tax_liability"].clip(lower=0.0)

    return df


def train_and_sample(
    model_name:         str,
    input_path:         Path,
    output_path:        Path,
    target_records:     int,
    discrete_cols:      list[str],
    fraud_rate_target:  float,
    epochs:             int = 300,
    batch_size:         int = 500,
) -> None:
    """
    Train a CTGAN model on the seed parquet and sample target_records rows.
    Applies post-constraints and enforces the target fraud rate.
    """
    import sys
    sys.path.insert(0, "/root")
    import os
    os.environ["MODAL_TASK_ID"] = "1"

    from config import RANDOM_SEED, FRAUD_RATE_TOLERANCE
    from utils import get_logger, make_dirs, write_parquet, enforce_fraud_rate

    log = get_logger(f"ctabgan_{model_name}", f"08_ctabgan_{model_name}.log")
    make_dirs(output_path.parent)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Seed file not found: {input_path}\n"
            "Run 06_generate_splits.py first."
        )

    log.info("Loading seed: %s", input_path)
    seed = pd.read_parquet(input_path)
    log.info(
        "Seed rows: %d  columns: %d  fraud_rate: %.4f",
        len(seed), len(seed.columns), seed["fraud_label"].mean(),
    )

    df_train = _prep_df(seed, DROP_COLS)

    # Only keep discrete_cols that actually exist after prep
    disc = [c for c in discrete_cols if c in df_train.columns]
    log.info(
        "Training CTGAN: epochs=%d  batch=%d  target=%d",
        epochs, batch_size, target_records,
    )

    from ctgan import CTGAN

    model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        embedding_dim=128,
        pac=10,
        verbose=True,
        enable_gpu=True, 
    )

    # CTGAN 0.9.0 does not expose a seed parameter on fit();
    # set both numpy and torch seeds for best-effort reproducibility
    import torch
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    model.fit(df_train, discrete_columns=disc)
    log.info("Fit complete. Sampling %d records...", target_records)

    synthetic = model.sample(target_records)
    log.info("Sampled: %d rows", len(synthetic))

    synthetic = _apply_post_constraints(synthetic, model_name)

    synthetic = enforce_fraud_rate(
        synthetic,
        target_rate=fraud_rate_target,
        tolerance=FRAUD_RATE_TOLERANCE,
        label_col="fraud_label",
    )

    write_parquet(synthetic, output_path)
    log.info(
        "Saved -> %s  final_fraud_rate=%.4f",
        output_path, synthetic["fraud_label"].mean(),
    )


# ── Modal functions ───────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="T4",
    cpu=4,
    memory=16_384,
    timeout=86_400,
    retries=0,
)
def train_w2():
    import sys, os
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"
    from config import SEED_W2, GAN_W2, FRAUD_RATE_W2
    train_and_sample(
        model_name        = "w2",
        input_path        = SEED_W2,
        output_path       = GAN_W2,
        target_records    = 1_800_000,
        discrete_cols     = DISCRETE_W2,
        fraud_rate_target = FRAUD_RATE_W2,
        epochs            = 300,
    )
    gan_vol.commit()
    logs_vol.commit()


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="T4",
    cpu=4,
    memory=16_384,
    timeout=86_400,
    retries=0,
)
def train_se():
    import sys, os
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"
    from config import SEED_SE, GAN_SE, FRAUD_RATE_SE
    train_and_sample(
        model_name        = "se",
        input_path        = SEED_SE,
        output_path       = GAN_SE,
        target_records    = 1_050_000,
        discrete_cols     = DISCRETE_SE,
        fraud_rate_target = FRAUD_RATE_SE,
        epochs            = 400,
    )
    gan_vol.commit()
    logs_vol.commit()


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="T4",
    cpu=4,
    memory=16_384,
    timeout=86_400,
    retries=0,
)
def train_itemizers():
    import sys, os
    sys.path.insert(0, "/root")
    os.environ["MODAL_TASK_ID"] = "1"
    from config import SEED_ITEMIZERS, GAN_ITEMIZERS, FRAUD_RATE_ITEMIZERS
    # Resolve path inside Modal context (MODAL_TASK_ID already set above)
    if not SEED_ITEMIZERS.exists():
        print(f"No itemizers seed found at {SEED_ITEMIZERS} — skipping.")
        return
    train_and_sample(
        model_name        = "itemizers",
        input_path        = SEED_ITEMIZERS,
        output_path       = GAN_ITEMIZERS,
        target_records    = 525_000,
        discrete_cols     = DISCRETE_ITEM,
        fraud_rate_target = FRAUD_RATE_ITEMIZERS,
        epochs            = 300,
    )
    gan_vol.commit()
    logs_vol.commit()


@app.local_entrypoint()
def main():
    print("Run one of:")
    print("  modal run 08_ctab_gan_runner.py::train_w2")
    print("  modal run 08_ctab_gan_runner.py::train_se")
    print("  modal run 08_ctab_gan_runner.py::train_itemizers")