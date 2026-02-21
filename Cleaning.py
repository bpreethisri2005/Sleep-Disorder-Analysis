import pandas as pd

# Load dataset
file_path = r"C:\Users\User\OneDrive\Documents\DataMining Proj\sleep_disorder_dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# ---------------------
# 1. Remove Duplicates
# ---------------------
df.drop_duplicates(inplace=True)

# ---------------------
# 2. Handle Missing Values
# ---------------------
# Drop rows where Patient_ID is missing
df.dropna(subset=["Patient_ID"], inplace=True)

# Fill numeric NaN with median
df["Age"].fillna(df["Age"].median(), inplace=True)
df["AHI_Score"].fillna(df["AHI_Score"].median(), inplace=True)
df["SaO2_Level"].fillna(df["SaO2_Level"].median(), inplace=True)

# Fill categorical NaN with mode
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Sleep_Disorder_Type"].fillna(df["Sleep_Disorder_Type"].mode()[0], inplace=True)

# ---------------------
# 3. Standardize Text Columns
# ---------------------
df["Gender"] = df["Gender"].str.strip().str.title()  # Male/Female

df["Sleep_Disorder_Type"] = df["Sleep_Disorder_Type"].str.strip().str.title()

# ---------------------
# 4. Convert Data Types
# ---------------------
df["Diagnosis_Confirmed"] = df["Diagnosis_Confirmed"].astype(int)

# ---------------------
# 5. Drop Irrelevant Columns (Optional)
# ---------------------
if "OCR_Extracted_Text" in df.columns:
    df.drop(columns=["OCR_Extracted_Text"], inplace=True)

# ---------------------
# 6. Save Cleaned Data
# ---------------------
cleaned_file = r"C:\Users\User\OneDrive\Documents\DM\cleaned_sleep_disorder_dataset.xlsx"
df.to_excel(cleaned_file, index=False)

print("Data cleaning complete! Cleaned file saved as:", cleaned_file)
