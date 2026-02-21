import pandas as pd
import random

# Load cleaned dataset
sleep_df = pd.read_excel(r"C:\Users\User\OneDrive\Documents\DataMining Proj\cleaned_sleep_disorder_dataset.xlsx")

# Create synthetic lifestyle dataset
lifestyle_data = {
    "Patient_ID": sleep_df["Patient_ID"],  # same patients
    "Smoking_Habit": [random.choice(["Yes", "No"]) for _ in range(len(sleep_df))],
    "Alcohol_Consumption": [random.choice(["Low", "Medium", "High"]) for _ in range(len(sleep_df))],
    "Exercise_Frequency": [random.randint(0, 7) for _ in range(len(sleep_df))],  # 0-7 days per week
    "Caffeine_Intake": [random.choice(["Low", "Medium", "High"]) for _ in range(len(sleep_df))],
    "Work_Shift": [random.choice(["Day", "Night", "Rotational"]) for _ in range(len(sleep_df))]
}

lifestyle_df = pd.DataFrame(lifestyle_data)

# Save lifestyle dataset
lifestyle_file = r"C:\Users\User\OneDrive\Documents\DataMining Proj\patient_lifestyle_data.xlsx"
lifestyle_df.to_excel(lifestyle_file, index=False)

print("Synthetic lifestyle dataset created and saved as:", lifestyle_file)
