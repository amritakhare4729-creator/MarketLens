import pandas as pd
import os


# ============================================================
# MARKETLENS - FINAL MODEL COMPARISON
# ============================================================

print("=" * 70)
print("MARKETLENS - FINAL MODEL COMPARISON")
print("=" * 70)


# ------------------------------------------------------------
# Model results collected from previous experiments
# ------------------------------------------------------------

results = [
    {
        "Model": "Random Forest Regressor",
        "Task": "Next-Day Close Price",
        "MAE": 36.7940,
        "RMSE": 120.1711,
        "R2": 0.9985,
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "ROC_AUC": None,
        "Baseline": 21.3812,
        "Primary_Metric": "MAE",
        "Verdict": "Worse than baseline"
    },

    {
        "Model": "Random Forest Regressor",
        "Task": "Next-Day Return",
        "MAE": 1.0998,
        "RMSE": 1.5423,
        "R2": -0.0610,
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "ROC_AUC": None,
        "Baseline": 1.0647,
        "Primary_Metric": "MAE",
        "Verdict": "Worse than baseline"
    },

    {
        "Model": "Improved Random Forest",
        "Task": "Next-Day Return",
        "MAE": 1.0788,
        "RMSE": 1.5115,
        "R2": -0.0348,
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "ROC_AUC": None,
        "Baseline": 1.0566,
        "Primary_Metric": "MAE",
        "Verdict": "Worse than baseline"
    },

    {
        "Model": "Random Forest Classifier",
        "Task": "UP / DOWN Direction",
        "MAE": None,
        "RMSE": None,
        "R2": None,
        "Accuracy": 48.77,
        "Precision": 46.68,
        "Recall": 46.79,
        "F1": 46.73,
        "ROC_AUC": 0.4900,
        "Baseline": 48.03,
        "Primary_Metric": "Accuracy",
        "Verdict": "Weak predictive performance"
    },

    {
        "Model": "Gradient Boosting",
        "Task": "UP / DOWN Direction",
        "MAE": None,
        "RMSE": None,
        "R2": None,
        "Accuracy": 49.23,
        "Precision": 47.93,
        "Recall": 65.88,
        "F1": 55.49,
        "ROC_AUC": 0.4922,
        "Baseline": 48.03,
        "Primary_Metric": "Accuracy",
        "Verdict": "Weak predictive performance"
    },

    {
        "Model": "Logistic Regression",
        "Task": "UP / DOWN Direction",
        "MAE": None,
        "RMSE": None,
        "R2": None,
        "Accuracy": 49.23,
        "Precision": 47.90,
        "Recall": 64.82,
        "F1": 55.09,
        "ROC_AUC": 0.4997,
        "Baseline": 48.03,
        "Primary_Metric": "Accuracy",
        "Verdict": "Weak predictive performance"
    }
]


# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(results)


# ------------------------------------------------------------
# Display complete comparison
# ------------------------------------------------------------

print("\nFINAL MODEL LEADERBOARD")
print("-" * 70)

display_columns = [
    "Model",
    "Task",
    "MAE",
    "RMSE",
    "R2",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "Baseline",
    "Verdict"
]

print(df[display_columns].to_string(index=False))


# ------------------------------------------------------------
# Best models
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST PERFORMING MODELS")
print("=" * 70)


classification_df = df[
    df["Task"] == "UP / DOWN Direction"
].copy()

best_accuracy = classification_df.loc[
    classification_df["Accuracy"].idxmax()
]

best_f1 = classification_df.loc[
    classification_df["F1"].idxmax()
]

best_auc = classification_df.loc[
    classification_df["ROC_AUC"].idxmax()
]


print(
    f"\nHighest Classification Accuracy:\n"
    f"{best_accuracy['Model']} - "
    f"{best_accuracy['Accuracy']:.2f}%"
)

print(
    f"\nHighest Classification F1 Score:\n"
    f"{best_f1['Model']} - "
    f"{best_f1['F1']:.2f}%"
)

print(
    f"\nHighest ROC-AUC:\n"
    f"{best_auc['Model']} - "
    f"{best_auc['ROC_AUC']:.4f}"
)


# ------------------------------------------------------------
# Regression comparison
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)

regression_df = df[
    df["MAE"].notna()
].copy()

print("\nLower MAE is better:\n")

for _, row in regression_df.iterrows():
    print(
        f"{row['Model']} | "
        f"{row['Task']} | "
        f"MAE: {row['MAE']:.4f} | "
        f"Baseline: {row['Baseline']:.4f} | "
        f"{row['Verdict']}"
    )


# ------------------------------------------------------------
# Classification comparison
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 70)

for _, row in classification_df.iterrows():

    difference = row["Accuracy"] - row["Baseline"]

    print(
        f"\n{row['Model']}"
    )

    print(
        f"Accuracy:  {row['Accuracy']:.2f}%"
    )

    print(
        f"Baseline:  {row['Baseline']:.2f}%"
    )

    print(
        f"Difference: {difference:+.2f} percentage points"
    )

    print(
        f"F1 Score:  {row['F1']:.2f}%"
    )

    print(
        f"ROC-AUC:   {row['ROC_AUC']:.4f}"
    )


# ------------------------------------------------------------
# Final recommendation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MARKETLENS MODEL DECISION")
print("=" * 70)

print(
    """
Based on all experiments, none of the tested models provides
a strong and reliable next-day stock prediction advantage.

The classification models perform close to random prediction,
with ROC-AUC values around 0.50.

The regression models also fail to outperform their respective
simple baselines.

Therefore, MarketLens should NOT claim highly accurate stock
price prediction.

The project should position machine learning as an
experimental decision-support component alongside historical
analysis, technical indicators, trends, volatility and
visual analytics.
"""
)


# ------------------------------------------------------------
# Save final comparison
# ------------------------------------------------------------

output_folder = "data/model_results"

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "final_model_comparison.csv"
)

df.to_csv(output_file, index=False)


print("=" * 70)
print(f"Final comparison saved to: {output_file}")
print("=" * 70)