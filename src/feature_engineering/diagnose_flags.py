# flag_combo_diagnostic.py
# Run standalone on any engineered parquet to see
# if weak flags add value in combination with strong ones

import pandas as pd
import numpy as np
from pathlib import Path

ENG = Path(r"D:\tax_evasion_synthetic_data\scripts_and_outputs\merged_splits\featured\engineered")

df = pd.read_parquet(ENG / "year_2019_engineered.parquet")

fraud = df["fraud_label"]
print(f"Base fraud rate: {fraud.mean():.4f}\n")

# Check if weak flags add lift ON TOP OF strong flags
print("Conditional fraud rates (weak flags on top of strong flag combos):")
print("=" * 70)

combos = [
    # Does flag_zero_tax add value when irs_high is already set?
    ("irs_high=1, zero_tax=1",
     (df["flag_irs_high"] == 1) & (df["flag_zero_tax_with_income"] == 1)),
    ("irs_high=1, zero_tax=0",
     (df["flag_irs_high"] == 1) & (df["flag_zero_tax_with_income"] == 0)),

    # Does flag_progressive_anomaly add on top of irs_high?
    ("irs_high=1, progressive=1",
     (df["flag_irs_high"] == 1) & (df["flag_progressive_anomaly"] == 1)),
    ("irs_high=1, progressive=0",
     (df["flag_irs_high"] == 1) & (df["flag_progressive_anomaly"] == 0)),

    # Does flag_low_withholding add on top of irs_high?
    ("irs_high=1, low_wh=1",
     (df["flag_irs_high"] == 1) & (df["flag_low_withholding"] == 1)),
    ("irs_high=1, low_wh=0",
     (df["flag_irs_high"] == 1) & (df["flag_low_withholding"] == 0)),

    # Does flag_impossible_deductions add on top of bank_deposit_extreme?
    ("bdr_extreme=1, impossible_ded=1",
     (df["flag_bank_deposit_extreme"] == 1) & (df["flag_impossible_deductions"] == 1)),
    ("bdr_extreme=1, impossible_ded=0",
     (df["flag_bank_deposit_extreme"] == 1) & (df["flag_impossible_deductions"] == 0)),

    # Triple combo — the holy grail
    ("irs_high=1, bdr_extreme=1, low_wh=1",
     (df["flag_irs_high"] == 1) &
     (df["flag_bank_deposit_extreme"] == 1) &
     (df["flag_low_withholding"] == 1)),

    ("irs_high=1, bdr_high=1, zero_tax=1",
     (df["flag_irs_high"] == 1) &
     (df["flag_bank_deposit_high"] == 1) &
     (df["flag_zero_tax_with_income"] == 1)),
]

for label, mask in combos:
    n     = mask.sum()
    if n == 0:
        print(f"  {label:<45} n=0")
        continue
    rate  = fraud[mask].mean()
    print(f"  {label:<45} n={n:>7,}  fraud={rate:.4f}  lift={rate/fraud.mean():.2f}x")

print()

# Check total_flag_count distribution
print("total_flag_count distribution:")
print("=" * 70)
for count in range(0, 9):
    mask = df["total_flag_count"] == count
    n    = mask.sum()
    if n == 0:
        continue
    rate = fraud[mask].mean()
    print(
        f"  count={count}  n={n:>8,} ({n/len(df)*100:>4.1f}%)  "
        f"fraud={rate:.4f}  lift={rate/fraud.mean():.2f}x"
    )

print()

# Check AUC of total_flag_count as a standalone predictor
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(fraud, df["total_flag_count"])
print(f"total_flag_count standalone AUC: {max(auc, 1-auc):.4f}")
print(f"flag_irs_high    standalone AUC: "
      f"{max(roc_auc_score(fraud, df['flag_irs_high']), 0.5):.4f}")
print(f"master_fraud_propensity AUC    : "
      f"{max(roc_auc_score(fraud, df['master_fraud_propensity']), 0.5):.4f}")