import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
print("MARKETLENS - STOCK DIRECTION CLASSIFICATION MODEL")
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

# Next-day return is already available in the dataset.
# UP = 1
# DOWN = 0

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

# Remove rows where next-day return cannot be calculated
df = df.dropna(subset=["Next_Day_Return"])

# ---------------------------------------------------------
# 3. FEATURES
# ---------------------------------------------------------

features = [
    # Original features
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

    # Advanced features
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

# Check required columns
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

print("\nCLEANING CLASSIFICATION DATA")
print("-" * 70)

X = df[features].copy()
y = df["Target_Direction"].copy()

# Replace infinity values
X = X.replace([np.inf, -np.inf], np.nan)

# Combine so rows stay aligned
model_data = pd.concat(
    [
        df[["Date", "Company", "Next_Day_Return"]],
        X,
        y.rename("Target_Direction")
    ],
    axis=1
)

missing_before = model_data[features].isna().sum().sum()

model_data = model_data.dropna(
    subset=features + ["Target_Direction"]
)

missing_after = model_data[features].isna().sum().sum()

print(f"Missing feature values before cleaning: {missing_before}")
print(f"Missing feature values after cleaning:  {missing_after}")
print(f"Rows available for modeling: {len(model_data)}")

# ---------------------------------------------------------
# 5. DATA VALIDATION
# ---------------------------------------------------------

print("\nDATA VALIDATION")
print("-" * 70)

remaining_missing = model_data[features].isna().sum().sum()

remaining_infinite = np.isinf(
    model_data[features].select_dtypes(include=[np.number])
).sum().sum()

print(f"Remaining missing values:   {remaining_missing}")
print(f"Remaining infinite values:  {remaining_infinite}")

if remaining_missing > 0 or remaining_infinite > 0:
    print("ERROR: Data validation failed.")
    raise SystemExit

print("Data validation successful!")

# ---------------------------------------------------------
# 6. CLASS DISTRIBUTION
# ---------------------------------------------------------

print("\nTARGET CLASS DISTRIBUTION")
print("-" * 70)

class_counts = model_data["Target_Direction"].value_counts()

down_count = class_counts.get(0, 0)
up_count = class_counts.get(1, 0)

total_count = down_count + up_count

print(f"DOWN (0): {down_count} ({down_count / total_count * 100:.2f}%)")
print(f"UP   (1): {up_count} ({up_count / total_count * 100:.2f}%)")

# ---------------------------------------------------------
# 7. TIME-BASED TRAIN/TEST SPLIT
# ---------------------------------------------------------

model_data = model_data.sort_values(
    ["Date", "Company"]
).reset_index(drop=True)

split_index = int(len(model_data) * 0.80)

train_data = model_data.iloc[:split_index]
test_data = model_data.iloc[split_index:]

X_train = train_data[features]
y_train = train_data["Target_Direction"]

X_test = test_data[features]
y_test = test_data["Target_Direction"]

print("\nTIME-BASED SPLIT")
print("-" * 70)

print(f"Training data: {len(X_train)} rows")
print(f"Testing data:  {len(X_test)} rows")
print(f"Split date:    {test_data['Date'].min().date()}")

# ---------------------------------------------------------
# 8. TRAIN RANDOM FOREST CLASSIFIER
# ---------------------------------------------------------

print("\nTraining Random Forest classification model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Model training completed!")

# ---------------------------------------------------------
# 9. PREDICTIONS
# ---------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 10. MODEL PERFORMANCE
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

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

try:
    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )
except ValueError:
    roc_auc = np.nan

print("\nCLASSIFICATION MODEL PERFORMANCE")
print("-" * 70)

print(f"Accuracy:   {accuracy * 100:.2f}%")
print(f"Precision:  {precision * 100:.2f}%")
print(f"Recall:     {recall * 100:.2f}%")
print(f"F1 Score:   {f1 * 100:.2f}%")
print(f"ROC-AUC:    {roc_auc:.4f}")

# ---------------------------------------------------------
# 11. BASELINE
# ---------------------------------------------------------

# Majority-class baseline
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
print(f"Actual DOWN   {cm[0,0]:6d}  {cm[0,1]:6d}")
print(f"Actual UP     {cm[1,0]:6d}  {cm[1,1]:6d}")

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
# 14. FEATURE IMPORTANCE
# ---------------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nTOP FEATURE IMPORTANCE")
print("-" * 70)

print(
    importance_df.head(15).to_string(
        index=False
    )
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
    "classification_predictions.csv"
)

importance_file = (
    "data/model_results/"
    "classification_feature_importance.csv"
)

comparison_file = (
    "data/model_results/"
    "classification_model_comparison.csv"
)

results.to_csv(
    prediction_file,
    index=False
)

importance_df.to_csv(
    importance_file,
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
    "Random Forest": [
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
print(importance_file)
print(comparison_file)

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL COMPLETED")
print("=" * 70)