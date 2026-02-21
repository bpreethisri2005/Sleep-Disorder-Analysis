import pandas as pd

# Load datasets
sleep_file = r"C:\Users\User\OneDrive\Documents\DataMining Proj\cleaned_sleep_disorder_dataset.xlsx"
lifestyle_file = r"C:\Users\User\OneDrive\Documents\DataMining Proj\patient_lifestyle_data.xlsx"

sleep_df = pd.read_excel(sleep_file)
lifestyle_df = pd.read_excel(lifestyle_file)

# ✅ Clean Patient_ID column to avoid mismatch
sleep_df["Patient_ID"] = sleep_df["Patient_ID"].astype(str).str.strip()
lifestyle_df["Patient_ID"] = lifestyle_df["Patient_ID"].astype(str).str.strip()

# Check if IDs match
print("Unique IDs in sleep dataset:", len(sleep_df["Patient_ID"].unique()))
print("Unique IDs in lifestyle dataset:", len(lifestyle_df["Patient_ID"].unique()))

# Check common IDs
common_ids = set(sleep_df["Patient_ID"]).intersection(set(lifestyle_df["Patient_ID"]))
print("Number of common Patient_IDs:", len(common_ids))

# Merge datasets only on common IDs
integrated_df = pd.merge(sleep_df, lifestyle_df, on="Patient_ID", how="inner")

# Save result
integrated_file = r"C:\Users\User\OneDrive\Documents\DataMining Proj\integrated_dataset.xlsx"
integrated_df.to_excel(integrated_file, index=False)

print("Integration complete! Rows after merge:", len(integrated_df))
