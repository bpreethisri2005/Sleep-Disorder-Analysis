import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- DATA LOADING -----------------------------
data = pd.read_excel("subset_reduced_dataset.xlsx", usecols="A:F")
print("✅ Dataset Loaded Successfully!\n")

# ----------------------------- DATA PREPROCESSING -----------------------------
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == "object":
        data[column] = le.fit_transform(data[column])

# Split dataset
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ----------------------------- CLASSIFIERS -----------------------------
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}
combined_cm = np.zeros((2, 2), dtype=int)

# ----------------------------- TRAINING & EVALUATION -----------------------------
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Handle binary or multiclass case
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = np.nan

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f_measure = f1_score(y_test, y_pred, average="macro")
    error_rate = 1 - accuracy
    specificity = tn / (tn + fp) if not np.isnan(tn) else np.nan
    sensitivity = tp / (tp + fn) if not np.isnan(tp) else np.nan

    results[name] = {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Error Rate": error_rate,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F-Measure": f_measure,
        "Confusion Matrix": cm,
    }

    # ---------------- Confusion Matrix (Individual) ----------------
    print(f"\n===== {name} =====")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Error Rate: {error_rate:.4f}")
    print(f"Sensitivity (Recall): {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"F-Measure: {f_measure:.4f}")
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Add to combined confusion matrix
    if cm.shape == (2, 2):
        combined_cm += cm

# ----------------------------- COMBINED CONFUSION MATRIX -----------------------------
print("\n===== Combined Confusion Matrix =====")
print(combined_cm)
plt.figure(figsize=(5, 4))
sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Purples')
plt.title("Combined Confusion Matrix (Decision Tree + Naive Bayes + KNN)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ----------------------------- ROC CURVE (COMBINED PLOT) -----------------------------
if len(np.unique(y)) == 2:
    plt.figure(figsize=(8, 6))
    for name, clf in classifiers.items():
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
        else:
            print(f"⚠️ {name} does not support predict_proba for ROC curve.")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curve (Decision Tree, Naive Bayes, KNN)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("\n ROC Curve skipped — dataset must be binary for combined ROC plotting.")

# ----------------------------- SUMMARY TABLE -----------------------------
summary_df = pd.DataFrame.from_dict(results, orient="index")
summary_df = summary_df.drop(columns=["Confusion Matrix"])
summary_df = summary_df.round(4)

print("\n===== CLASSIFIER PERFORMANCE SUMMARY =====")
print(summary_df)

# Save summary to Excel
summary_df.to_excel("classifier_summary.xlsx")
print("\n Summary saved as 'classifier_summary.xlsx'")

# ----------------------------- BEST CLASSIFIER COMPARISON -----------------------------
best_acc = summary_df["Accuracy"].idxmax()
best_f1 = summary_df["F-Measure"].idxmax()

print("\n BEST CLASSIFIER RESULTS ")
print(f"Based on Accuracy: {best_acc} with {summary_df.loc[best_acc, 'Accuracy']:.4f}")
print(f"Based on F-Measure: {best_f1} with {summary_df.loc[best_f1, 'F-Measure']:.4f}")

if best_acc == best_f1:
    print(f"\n The overall best performing classifier is {best_acc}, consistently leading in both Accuracy and F-measure.")
else:
    print(f"\n The top performer for accuracy is {best_acc}, while {best_f1} excels in F-measure, indicating strong balance across metrics.")
