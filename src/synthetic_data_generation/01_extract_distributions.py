# Run this as a quick Modal inspection or locally if you can access the volume
import pandas as pd

emp = pd.read_csv("/final_dataset/reference/employment_links.csv")
persons = pd.read_csv("/final_dataset/reference/persons.csv")

w2_persons = persons[persons["taxpayer_type"].isin(["pure_w2", "w2_with_side_biz"])]
covered = emp[emp["person_id"].isin(w2_persons["person_id"])]["person_id"].nunique()

print(f"W2 persons:    {len(w2_persons):,}")
print(f"With emp link: {covered:,}")
print(f"Coverage:      {covered/len(w2_persons):.1%}")