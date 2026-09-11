import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 70)
print("MARKETLENS - PER-COMPANY CLASSIFICATION ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD CLASSIFICATION RESULTS
# ---------------------------------------------------------

file_path = "data/model_results/classification_predictions.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

print(f"Loaded predictions: {len(df)} rows")
print(f"Companies: {df['Company'].nunique()}")

# ---------------------------------------------------------
# 2. VERIFY REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = [
    "Date",
    "Company",
    "Actual_Direction",
    "Predicted_Direction",
    "UP_Probability",
    "Correct"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    for column in missing_columns:
        print(column)
    raise SystemExit

# ---------------------------------------------------------
# 3. CONVERT DIRECTIONS TO NUMERIC
# ---------------------------------------------------------

df["Actual"] = np.where(
    df["Actual_Direction"] == "UP",
    1,
    0
)

df["Predicted"] = np.where(
    df["Predicted_Direction"] == "UP",
    1,
    0
)

# ---------------------------------------------------------
# 4. ANALYZE EACH COMPANY
# ---------------------------------------------------------

company_results = []

for company in sorted(df["Company"].unique()):

    company_df = df[df["Company"] == company]

    y_actual = company_df["Actual"]
    y_predicted = company_df["Predicted"]

    accuracy = accuracy_score(
        y_actual,
        y_predicted
    )

    precision = precision_score(
        y_actual,
        y_predicted,
        zero_division=0
    )

    recall = recall_score(
        y_actual,
        y_predicted,
        zero_division=0
    )

    f1 = f1_score(
        y_actual,
        y_predicted,
        zero_division=0
    )

    up_actual = (y_actual == 1).sum()
    down_actual = (y_actual == 0).sum()

    up_predicted = (y_predicted == 1).sum()
    down_predicted = (y_predicted == 0).sum()

    company_results.append({
        "Company": company,
        "Test_Rows": len(company_df),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Actual_UP": up_actual,
        "Actual_DOWN": down_actual,
        "Predicted_UP": up_predicted,
        "Predicted_DOWN": down_predicted
    })

company_results_df = pd.DataFrame(company_results)

# ---------------------------------------------------------
# 5. SORT BY ACCURACY
# ---------------------------------------------------------

company_results_df = company_results_df.sort_values(
    "Accuracy",
    ascending=False
).reset_index(drop=True)

print("\nPER-COMPANY PERFORMANCE")
print("-" * 70)

display_df = company_results_df.copy()

display_df["Accuracy"] = (
    display_df["Accuracy"] * 100
).round(2)

display_df["Precision"] = (
    display_df["Precision"] * 100
).round(2)

display_df["Recall"] = (
    display_df["Recall"] * 100
).round(2)

display_df["F1_Score"] = (
    display_df["F1_Score"] * 100
).round(2)

print(display_df.to_string(index=False))

# ---------------------------------------------------------
# 6. BEST AND WORST COMPANIES
# ---------------------------------------------------------

best_company = company_results_df.iloc[0]

worst_company = company_results_df.iloc[-1]

print("\nBEST-PERFORMING COMPANY")
print("-" * 70)

print(
    f"Company:  {best_company['Company']}"
)

print(
    f"Accuracy: {best_company['Accuracy'] * 100:.2f}%"
)

print(
    f"F1 Score: {best_company['F1_Score'] * 100:.2f}%"
)

print("\nWORST-PERFORMING COMPANY")
print("-" * 70)

print(
    f"Company:  {worst_company['Company']}"
)

print(
    f"Accuracy: {worst_company['Accuracy'] * 100:.2f}%"
)

print(
    f"F1 Score: {worst_company['F1_Score'] * 100:.2f}%"
)

# ---------------------------------------------------------
# 7. COMPANIES ABOVE 50%
# ---------------------------------------------------------

above_50 = company_results_df[
    company_results_df["Accuracy"] > 0.50
]

print("\nCOMPANIES ABOVE 50% ACCURACY")
print("-" * 70)

if len(above_50) == 0:
    print("No company achieved accuracy above 50%.")
else:
    for _, row in above_50.iterrows():
        print(
            f"{row['Company']}: "
            f"{row['Accuracy'] * 100:.2f}%"
        )

# ---------------------------------------------------------
# 8. COMPANIES ABOVE 55%
# ---------------------------------------------------------

above_55 = company_results_df[
    company_results_df["Accuracy"] > 0.55
]

print("\nCOMPANIES ABOVE 55% ACCURACY")
print("-" * 70)

if len(above_55) == 0:
    print("No company achieved accuracy above 55%.")
else:
    for _, row in above_55.iterrows():
        print(
            f"{row['Company']}: "
            f"{row['Accuracy'] * 100:.2f}%"
        )

# ---------------------------------------------------------
# 9. OVERALL COMPANY STATISTICS
# ---------------------------------------------------------

average_company_accuracy = (
    company_results_df["Accuracy"].mean()
)

median_company_accuracy = (
    company_results_df["Accuracy"].median()
)

best_accuracy = (
    company_results_df["Accuracy"].max()
)

worst_accuracy = (
    company_results_df["Accuracy"].min()
)

print("\nCOMPANY-LEVEL SUMMARY")
print("-" * 70)

print(
    f"Average company accuracy: "
    f"{average_company_accuracy * 100:.2f}%"
)

print(
    f"Median company accuracy:  "
    f"{median_company_accuracy * 100:.2f}%"
)

print(
    f"Best accuracy:            "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Worst accuracy:           "
    f"{worst_accuracy * 100:.2f}%"
)

# ---------------------------------------------------------
# 10. SAVE RESULTS
# ---------------------------------------------------------

import os

os.makedirs(
    "data/model_results",
    exist_ok=True
)

output_file = (
    "data/model_results/"
    "company_classification_performance.csv"
)

company_results_df.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")
print(output_file)

# ---------------------------------------------------------
# 11. COMPLETION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PER-COMPANY ANALYSIS COMPLETED")
print("=" * 70)