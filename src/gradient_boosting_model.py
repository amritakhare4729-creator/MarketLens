import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
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
print("MARKETLENS - GRADIENT BOOSTING CLASSIFICATION MODEL")
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

df["Target_Direction"] = np.where(
    df["Next_Day_Return"] > 0,
    1,
    0
)

df = df.dropna(
    subset=["Next_Day_Return"]
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
# 4. CLEAN DATA
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
# 5. CREATE TARGET AFTER CLEANING
# ---------------------------------------------------------

model_data["Target_Direction"] = np.where(
    model_data["Next_Day_Return"] > 0,
    1,
    0
)

# ---------------------------------------------------------
# 6. VALIDATION
# ---------------------------------------------------------

print("\nDATA VALIDATION")
print("-" * 70)

missing_values = (
    model_data[features]
    .isna()
    .sum()
    .sum()
)

infinite_values = np.isinf(
    model_data[features]
    .select_dtypes(include=[np.number])
).sum().sum()

print(
    f"Remaining missing values:  "
    f"{missing_values}"
)

print(
    f"Remaining infinite values: "
    f"{infinite_values}"
)

if missing_values > 0 or infinite_values > 0:
    print("ERROR: Data validation failed.")
    raise SystemExit

print("Data validation successful!")

# ---------------------------------------------------------
# 7. TIME-BASED SPLIT
# ---------------------------------------------------------

model_data = model_data.sort_values(
    ["Date", "Company"]
).reset_index(drop=True)

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
# 8. TRAIN GRADIENT BOOSTING MODEL
# ---------------------------------------------------------

print("\nTraining HistGradientBoosting model...")

model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Model training completed!")

# ---------------------------------------------------------
# 9. PREDICTIONS
# ---------------------------------------------------------

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]

# ---------------------------------------------------------
# 10. PERFORMANCE
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

print("\nGRADIENT BOOSTING PERFORMANCE")
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
# 11. BASELINE
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
# 12. CONFUSION MATRIX
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
# 13. CLASSIFICATION REPORT
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
# 14. PREDICTION RESULTS
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
# 15. SAVE RESULTS
# ---------------------------------------------------------

import os

os.makedirs(
    "data/model_results",
    exist_ok=True
)

prediction_file = (
    "data/model_results/"
    "gradient_boosting_predictions.csv"
)

comparison_file = (
    "data/model_results/"
    "gradient_boosting_model_comparison.csv"
)

results.to_csv(
    prediction_file,
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
    "Gradient Boosting": [
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
# 16. SAMPLE PREDICTIONS
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
# 17. SAVE FEATURE DATA FOR LATER ANALYSIS
# ---------------------------------------------------------

# HistGradientBoosting does not expose standard
# feature_importances_, so we will analyze feature
# importance later using permutation importance.

# ---------------------------------------------------------
# 18. COMPLETION
# ---------------------------------------------------------

print("\nResults saved to:")
print(prediction_file)
print(comparison_file)

print("\n" + "=" * 70)
print("GRADIENT BOOSTING MODEL COMPLETED")
print("=" * 70)