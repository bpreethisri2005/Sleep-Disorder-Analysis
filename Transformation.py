import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load integrated dataset
integrated_file = r"C:\Users\User\OneDrive\Documents\DataMining Proj\integrated_dataset.xlsx"
df = pd.read_excel(integrated_file)

# ---------------------
# 1. Normalize Numeric Columns
# ---------------------
scaler = MinMaxScaler()
numeric_cols = ["Age", "AHI_Score", "SaO2_Level"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ---------------------
# 2. Encode Categorical Variables
# ---------------------
label_enc = LabelEncoder()
categorical_cols = ["Gender", "Sleep_Disorder_Type", "Smoking_Habit", "Alcohol_Consumption", "Caffeine_Intake", "Work_Shift"]

for col in categorical_cols:
    df[col] = label_enc.fit_transform(df[col].astype(str))

# ---------------------
# 3. Create Age Group Feature
# ---------------------
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Young", "Middle", "Old"]
)

# ---------------------
# 4. Save Transformed Dataset
# ---------------------
transformed_file = r"C:\Users\User\OneDrive\Documents\DataMining Proj\transformed_dataset.xlsx"
df.to_excel(transformed_file, index=False)

print("Data transformation complete! Transformed file saved as:", transformed_file)
