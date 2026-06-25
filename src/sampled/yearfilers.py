import pandas as pd

df = pd.read_parquet(r"D:\projiikkkkttttt\data\by_state\california.parquet")

PID  = "person_id"
YEAR = "tax_year"
COH  = "entry_cohort"

years_per_person = df.groupby(PID)[YEAR].nunique()
one_year_pids    = years_per_person[years_per_person == 1].index

one_year_df = df[df[PID].isin(one_year_pids)]

print("=== 1-year filers: entry_cohort breakdown ===")
print(one_year_df[COH].value_counts(dropna=False))

print("\n=== 1-year filers: which years do they appear in? ===")
print(one_year_df[YEAR].value_counts().sort_index())

print("\n=== 1-year filers: fraud rate ===")
print(f"  {one_year_df['fraud_label'].mean():.4f}")

print("\n=== Multi-year filers: fraud rate ===")
multi_year_pids = years_per_person[years_per_person > 1].index
multi_df = df[df[PID].isin(multi_year_pids)]
print(f"  {multi_df['fraud_label'].mean():.4f}")

print("\n=== 1-year filers: taxpayer_type breakdown ===")
print(one_year_df["taxpayer_type"].value_counts(normalize=True))

print("\n=== Multi-year filers: taxpayer_type breakdown ===")
print(multi_df["taxpayer_type"].value_counts(normalize=True))