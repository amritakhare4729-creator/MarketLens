import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("MARKETLENS - LOGISTIC REGRESSION CLASSIFICATION MODEL")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

file_path = "data/advanced_features.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

print(f"Loaded dataset: {len(df)} rows")
print(f"Companies: {df['Company'].nunique()}")

# ---------------------------------------------------------
# 2. CREATE TARGET
# ---------------------------------------------------------

if "Next_Day_Return" not in df.columns:
    df["Next_Day_Return"] = (
        df.groupby("Company")["Close"].shift(-1)
        / df["Close"] - 1
    ) * 100

df = df.dropna(
    subset=["Next_Day_Return"]
)

df["Target_Direction"] = np.where(
    df["Next_Day_Return"] > 0,
    1,
    0
)

# ---------------------------------------------------------
# 3. FEATURES
# ---------------------------------------------------------

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "Price_Change",
    "MA_20",
    "MA_50",
    "Volatility_20",
    "High_Low_Range",
    "High_Low_Range_Pct",
    "Cumulative_Return",
    "Return_1D",
    "Return_3D",
    "Return_5D",
    "Return_10D",
    "Momentum_5D",
    "Momentum_10D",
    "Price_vs_MA20",
    "Price_vs_MA50",
    "Volume_Change",
    "Volume_MA20",
    "Volume_Ratio",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Volatility_5D",
    "Volatility_10D",
    "Open_Close_Change",
    "High_Low_Pct"
]

missing_columns = [
    column for column in features
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    for column in missing_columns:
        print(column)
    raise SystemExit

# ---------------------------------------------------------
# 4. PREPARE MODEL DATA
# ---------------------------------------------------------

print("\nCLEANING DATA")
print("-" * 70)

model_data = df[
    ["Date", "Company", "Next_Day_Return"]
    + features
].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

missing_before = model_data[features].isna().sum().sum()

model_data = model_data.dropna(
    subset=features + ["Next_Day_Return"]
)

missing_after = model_data[features].isna().sum().sum()

print(
    f"Missing feature values before cleaning: "
    f"{missing_before}"
)

print(
    f"Missing feature values after cleaning:  "
    f"{missing_after}"
)

print(
    f"Rows available for modeling: "
    f"{len(model_data)}"
)

# ---------------------------------------------------------
# 5. VALIDATION
# ---------------------------------------------------------

print("\nDATA VALIDATION")
print("-" * 70)

remaining_missing = (
    model_data[features]
    .isna()
    .sum()
    .sum()
)

remaining_infinite = np.isinf(
    model_data[features]
    .select_dtypes(include=[np.number])
).sum().sum()

print(
    f"Remaining missing values:  "
    f"{remaining_missing}"
)

print(
    f"Remaining infinite values: "
    f"{remaining_infinite}"
)

if remaining_missing > 0 or remaining_infinite > 0:
    print("ERROR: Data validation failed.")
    raise SystemExit

print("Data validation successful!")

# ---------------------------------------------------------
# 6. TIME-BASED SPLIT
# ---------------------------------------------------------

model_data = model_data.sort_values(
    ["Date", "Company"]
).reset_index(drop=True)

model_data["Target_Direction"] = np.where(
    model_data["Next_Day_Return"] > 0,
    1,
    0
)

split_index = int(
    len(model_data) * 0.80
)

train_data = model_data.iloc[:split_index]
test_data = model_data.iloc[split_index:]

X_train = train_data[features]
y_train = train_data["Target_Direction"]

X_test = test_data[features]
y_test = test_data["Target_Direction"]

print("\nTIME-BASED SPLIT")
print("-" * 70)

print(
    f"Training data: {len(X_train)} rows"
)

print(
    f"Testing data:  {len(X_test)} rows"
)

print(
    f"Split date:    "
    f"{test_data['Date'].min().date()}"
)

# ---------------------------------------------------------
# 7. TRAIN LOGISTIC REGRESSION
# ---------------------------------------------------------

print("\nTraining Logistic Regression model...")

# StandardScaler is important because the features
# have very different numerical scales.

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            C=0.1,
            random_state=42
        )
    )
])

model.fit(
    X_train,
    y_train
)

print("Model training completed!")

# ---------------------------------------------------------
# 8. PREDICTIONS
# ---------------------------------------------------------

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]

# ---------------------------------------------------------
# 9. PERFORMANCE
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nLOGISTIC REGRESSION PERFORMANCE")
print("-" * 70)

print(
    f"Accuracy:   {accuracy * 100:.2f}%"
)

print(
    f"Precision:  {precision * 100:.2f}%"
)

print(
    f"Recall:     {recall * 100:.2f}%"
)

print(
    f"F1 Score:   {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC:    {roc_auc:.4f}"
)

# ---------------------------------------------------------
# 10. BASELINE
# ---------------------------------------------------------

majority_class = y_train.mode()[0]

baseline_predictions = np.full(
    len(y_test),
    majority_class
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("\nBASELINE PERFORMANCE")
print("-" * 70)

print(
    f"Majority class: "
    f"{'UP' if majority_class == 1 else 'DOWN'}"
)

print(
    f"Baseline Accuracy: "
    f"{baseline_accuracy * 100:.2f}%"
)

# ---------------------------------------------------------
# 11. CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nCONFUSION MATRIX")
print("-" * 70)

print("                Predicted")
print("              DOWN     UP")

print(
    f"Actual DOWN   "
    f"{cm[0, 0]:6d}  "
    f"{cm[0, 1]:6d}"
)

print(
    f"Actual UP     "
    f"{cm[1, 0]:6d}  "
    f"{cm[1, 1]:6d}"
)

# ---------------------------------------------------------
# 12. CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["DOWN", "UP"],
        zero_division=0
    )
)

# ---------------------------------------------------------
# 13. MODEL COMPARISON
# ---------------------------------------------------------

accuracy_difference = (
    accuracy - baseline_accuracy
) * 100

print("\nMODEL VS BASELINE")
print("-" * 70)

print(
    f"Accuracy difference: "
    f"{accuracy_difference:+.2f} percentage points"
)

if accuracy > baseline_accuracy:
    print(
        "Logistic Regression beats the baseline "
        "on accuracy."
    )
else:
    print(
        "Logistic Regression does NOT beat the "
        "baseline on accuracy."
    )

if roc_auc > 0.50:
    print(
        "ROC-AUC is above 0.50, indicating "
        "some ranking signal."
    )
else:
    print(
        "ROC-AUC is at or below 0.50, indicating "
        "very weak predictive signal."
    )

# ---------------------------------------------------------
# 14. FEATURE COEFFICIENTS
# ---------------------------------------------------------

classifier = model.named_steps["classifier"]

coefficients = classifier.coef_[0]

coefficient_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients,
    "Absolute_Coefficient": np.abs(coefficients)
})

coefficient_df = coefficient_df.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print("\nTOP FEATURE COEFFICIENTS")
print("-" * 70)

print(
    coefficient_df[
        [
            "Feature",
            "Coefficient"
        ]
    ].head(15).to_string(index=False)
)

# ---------------------------------------------------------
# 15. PREDICTION RESULTS
# ---------------------------------------------------------

results = test_data[
    ["Date", "Company", "Next_Day_Return"]
].copy()

results["Actual_Direction"] = np.where(
    y_test.values == 1,
    "UP",
    "DOWN"
)

results["Predicted_Direction"] = np.where(
    y_pred == 1,
    "UP",
    "DOWN"
)

results["UP_Probability"] = y_probability

results["Correct"] = (
    results["Actual_Direction"]
    == results["Predicted_Direction"]
)

# ---------------------------------------------------------
# 16. SAVE RESULTS
# ---------------------------------------------------------

import os

os.makedirs(
    "data/model_results",
    exist_ok=True
)

prediction_file = (
    "data/model_results/"
    "logistic_regression_predictions.csv"
)

coefficient_file = (
    "data/model_results/"
    "logistic_regression_coefficients.csv"
)

comparison_file = (
    "data/model_results/"
    "logistic_regression_model_comparison.csv"
)

results.to_csv(
    prediction_file,
    index=False
)

coefficient_df.to_csv(
    coefficient_file,
    index=False
)

comparison_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Baseline Accuracy"
    ],
    "Logistic Regression": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        np.nan
    ],
    "Baseline": [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        baseline_accuracy
    ]
})

comparison_df.to_csv(
    comparison_file,
    index=False
)

# ---------------------------------------------------------
# 17. SAMPLE PREDICTIONS
# ---------------------------------------------------------

print("\nSAMPLE PREDICTIONS")
print("-" * 70)

print(
    results[
        [
            "Date",
            "Company",
            "Actual_Direction",
            "Predicted_Direction",
            "UP_Probability",
            "Correct"
        ]
    ].head(10).to_string(index=False)
)

# ---------------------------------------------------------
# 18. COMPLETION
# ---------------------------------------------------------

print("\nResults saved to:")
print(prediction_file)
print(coefficient_file)
print(comparison_file)

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION MODEL COMPLETED")
print("=" * 70)