import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("MARKETLENS - IMPROVED RETURN PREDICTION MODEL")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD ADVANCED FEATURES
# ---------------------------------------------------------

df = pd.read_csv("data/advanced_features.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Company", "Date"]
).reset_index(drop=True)

print(f"Loaded dataset: {len(df)} rows")
print(f"Companies: {df['Company'].nunique()}")

# ---------------------------------------------------------
# 2. DEFINE FEATURES
# ---------------------------------------------------------

features = [
    # Original price features
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    # Original indicators
    "Daily_Return",
    "Price_Change",
    "MA_20",
    "MA_50",
    "Volatility_20",
    "High_Low_Range",
    "High_Low_Range_Pct",
    "Cumulative_Return",

    # Advanced return features
    "Return_1D",
    "Return_3D",
    "Return_5D",
    "Return_10D",

    # Momentum
    "Momentum_5D",
    "Momentum_10D",

    # Moving-average relationships
    "Price_vs_MA20",
    "Price_vs_MA50",

    # Volume
    "Volume_Change",
    "Volume_MA20",
    "Volume_Ratio",

    # Technical indicators
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",

    # Volatility
    "Volatility_5D",
    "Volatility_10D",

    # Price movement
    "Open_Close_Change",
    "High_Low_Pct"
]

target = "Next_Day_Return"

# ---------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = features + [target]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print()
    print("ERROR: The following columns are missing:")
    print(missing_columns)
    raise SystemExit

# ---------------------------------------------------------
# 4. PREPARE MODEL DATA
# ---------------------------------------------------------

model_data = df[
    ["Date", "Company"] + features + [target]
].copy()

# ---------------------------------------------------------
# 5. REPLACE INFINITE VALUES
# ---------------------------------------------------------

print()
print("CLEANING MODEL DATA")
print("-" * 70)

infinite_count = np.isinf(
    model_data[features + [target]]
    .select_dtypes(include=np.number)
).sum().sum()

print(f"Infinite values found: {infinite_count}")

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

# ---------------------------------------------------------
# 6. REMOVE MISSING VALUES
# ---------------------------------------------------------

missing_before = model_data[
    features + [target]
].isna().sum().sum()

print(f"Missing values before cleaning: {missing_before}")

model_data = model_data.dropna().copy()

missing_after = model_data[
    features + [target]
].isna().sum().sum()

print(f"Missing values after cleaning:  {missing_after}")

print(f"Rows available for modeling: {len(model_data)}")

# ---------------------------------------------------------
# 7. FINAL DATA VALIDATION
# ---------------------------------------------------------

print()
print("DATA VALIDATION")
print("-" * 70)

numeric_data = model_data[
    features + [target]
].select_dtypes(include=np.number)

remaining_infinite = np.isinf(
    numeric_data
).sum().sum()

remaining_missing = numeric_data.isna().sum().sum()

print(f"Remaining missing values:   {remaining_missing}")
print(f"Remaining infinite values:  {remaining_infinite}")

if remaining_missing > 0 or remaining_infinite > 0:
    print()
    print("ERROR: Invalid values remain in the dataset.")
    raise SystemExit

print("Data validation successful!")

# ---------------------------------------------------------
# 8. TIME-BASED TRAIN/TEST SPLIT
# ---------------------------------------------------------

split_date = model_data["Date"].quantile(0.80)

train_data = model_data[
    model_data["Date"] <= split_date
].copy()

test_data = model_data[
    model_data["Date"] > split_date
].copy()

print()
print("TIME-BASED SPLIT")
print("-" * 70)

print(f"Training data: {len(train_data)} rows")
print(f"Testing data:  {len(test_data)} rows")
print(f"Split date:    {split_date.date()}")

# ---------------------------------------------------------
# 9. PREPARE X AND Y
# ---------------------------------------------------------

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

# ---------------------------------------------------------
# 10. TRAIN RANDOM FOREST
# ---------------------------------------------------------

print()
print("Training improved Random Forest model...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model training completed!")

# ---------------------------------------------------------
# 11. MAKE PREDICTIONS
# ---------------------------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------------------------
# 12. RANDOM FOREST PERFORMANCE
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print()
print("IMPROVED MODEL PERFORMANCE")
print("-" * 70)

print(f"MAE:  {mae:.4f}%")
print(f"RMSE: {rmse:.4f}%")
print(f"R²:   {r2:.4f}")

# ---------------------------------------------------------
# 13. ZERO-RETURN BASELINE
# ---------------------------------------------------------

# Baseline assumption:
# Tomorrow's return = 0%

baseline_pred = np.zeros(
    len(y_test)
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_pred
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_pred
)

print()
print("BASELINE PERFORMANCE")
print("-" * 70)

print(f"Baseline MAE:  {baseline_mae:.4f}%")
print(f"Baseline RMSE: {baseline_rmse:.4f}%")
print(f"Baseline R²:   {baseline_r2:.4f}")

# ---------------------------------------------------------
# 14. MODEL IMPROVEMENT
# ---------------------------------------------------------

mae_improvement = (
    (baseline_mae - mae)
    / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - rmse)
    / baseline_rmse
) * 100

print()
print("MODEL IMPROVEMENT")
print("-" * 70)

print(
    f"MAE improvement:  {mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: {rmse_improvement:.2f}%"
)

# ---------------------------------------------------------
# 15. DIRECTIONAL PREDICTION
# ---------------------------------------------------------

actual_direction = np.where(
    y_test >= 0,
    "UP",
    "DOWN"
)

predicted_direction = np.where(
    y_pred >= 0,
    "UP",
    "DOWN"
)

direction_accuracy = (
    actual_direction == predicted_direction
).mean() * 100

print()
print("DIRECTIONAL PERFORMANCE")
print("-" * 70)

print(
    f"Directional Accuracy: "
    f"{direction_accuracy:.2f}%"
)

# ---------------------------------------------------------
# 16. FEATURE IMPORTANCE
# ---------------------------------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

print()
print("TOP FEATURE IMPORTANCE")
print("-" * 70)

print(
    importance.head(15).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# 17. CREATE PREDICTION DATASET
# ---------------------------------------------------------

predictions = test_data[
    ["Date", "Company"]
].copy()

predictions["Actual_Return"] = (
    y_test.values
)

predictions["Predicted_Return"] = (
    y_pred
)

predictions["Prediction_Error"] = (
    predictions["Actual_Return"]
    - predictions["Predicted_Return"]
)

predictions["Actual_Direction"] = np.where(
    predictions["Actual_Return"] >= 0,
    "UP",
    "DOWN"
)

predictions["Predicted_Direction"] = np.where(
    predictions["Predicted_Return"] >= 0,
    "UP",
    "DOWN"
)

# ---------------------------------------------------------
# 18. SAVE PREDICTIONS
# ---------------------------------------------------------

predictions.to_csv(
    "data/model_results/improved_return_predictions.csv",
    index=False
)

# ---------------------------------------------------------
# 19. SAVE FEATURE IMPORTANCE
# ---------------------------------------------------------

importance.to_csv(
    "data/model_results/improved_return_feature_importance.csv",
    index=False
)

# ---------------------------------------------------------
# 20. SAVE MODEL COMPARISON
# ---------------------------------------------------------

comparison = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2 Score",
        "Directional Accuracy"
    ],
    "Baseline": [
        baseline_mae,
        baseline_rmse,
        baseline_r2,
        50.0
    ],
    "Improved Random Forest": [
        mae,
        rmse,
        r2,
        direction_accuracy
    ]
})

comparison.to_csv(
    "data/model_results/improved_model_comparison.csv",
    index=False
)

# ---------------------------------------------------------
# 21. SAMPLE PREDICTIONS
# ---------------------------------------------------------

print()
print("SAMPLE PREDICTIONS")
print("-" * 70)

print(
    predictions.head(10).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# 22. FILE OUTPUT
# ---------------------------------------------------------

print()
print("Results saved to:")
print(
    "data/model_results/"
    "improved_return_predictions.csv"
)

print(
    "data/model_results/"
    "improved_return_feature_importance.csv"
)

print(
    "data/model_results/"
    "improved_model_comparison.csv"
)

print()
print("=" * 70)
print("IMPROVED RETURN MODEL COMPLETED")
print("=" * 70)