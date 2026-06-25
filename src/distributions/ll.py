import sys
sys.path.insert(0, ".")

# Test config imports
from config import (
    MACRO_SHOCKS,
    STANDARD_DEDUCTIONS,
    SS_WAGE_BASE,
    INCOME_STREAM_RULES,
    ZONE_INCOME_MULTIPLIER,
    ZONE_HOMEOWNERSHIP_RATE,
    TAX_BRACKETS,
)
print("config OK")
print(f"  MACRO_SHOCKS years:      {list(MACRO_SHOCKS.keys())}")
print(f"  SS_WAGE_BASE years:      {list(SS_WAGE_BASE.keys())}")
print(f"  STANDARD_DEDUCTIONS yrs: {list(STANDARD_DEDUCTIONS.keys())}")
print(f"  INCOME_STREAM_RULES:     {list(INCOME_STREAM_RULES.keys())}")
print(f"  ZONE_INCOME_MULTIPLIER:  {ZONE_INCOME_MULTIPLIER}")
print(f"  ZONE_HOMEOWNERSHIP_RATE: {ZONE_HOMEOWNERSHIP_RATE}")
print(f"  TAX_BRACKETS years:      {list(TAX_BRACKETS.keys())}")

# Test utils import
from utils import compute_tax_liability
print("utils OK")
print(f"  compute_tax_liability(50000, 2023, 'single') = "
      f"{compute_tax_liability(50000, 2023, 'single')}")
print(f"  compute_tax_liability(50000, 2023, 'married_joint') = "
      f"{compute_tax_liability(50000, 2023, 'married_joint')}")

print("\nAll imports OK. Safe to run modal run 05_generate_panels.py")