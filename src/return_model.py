import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("MARKETLENS - NEXT-DAY RETURN PREDICTION")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("data/stock_features.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Company", "Date"]
).reset_index(drop=True)

print(f"Loaded dataset: {len(df)} rows")
print(f"Companies: {df['Company'].nunique()}")

# ---------------------------------------------------------
# 2. CREATE NEXT-DAY RETURN TARGET
# ---------------------------------------------------------

df["Next_Day_Return"] = (
    df.groupby("Company")["Close"]
    .shift(-1)
    / df["Close"]
    - 1
) * 100

# ---------------------------------------------------------
# 3. SELECT FEATURES
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
    "Cumulative_Return"
]

target = "Next_Day_Return"

# ---------------------------------------------------------
# 4. REMOVE MISSING VALUES
# ---------------------------------------------------------

model_data = df[
    ["Date", "Company"] + features + [target]
].dropna().copy()

print(f"Rows available for modeling: {len(model_data)}")

# ---------------------------------------------------------
# 5. TIME-BASED TRAIN/TEST SPLIT
# ---------------------------------------------------------

split_date = model_data["Date"].quantile(0.80)

train_data = model_data[
    model_data["Date"] <= split_date
]

test_data = model_data[
    model_data["Date"] > split_date
]

print()
print("TIME-BASED SPLIT")
print("-" * 70)
print(f"Training data: {len(train_data)} rows")
print(f"Testing data:  {len(test_data)} rows")
print(f"Split date:    {split_date.date()}")

# ---------------------------------------------------------
# 6. PREPARE X AND Y
# ---------------------------------------------------------

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

# ---------------------------------------------------------
# 7. TRAIN RANDOM FOREST
# ---------------------------------------------------------

print()
print("Training Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

# ---------------------------------------------------------
# 8. MAKE PREDICTIONS
# ---------------------------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------------------------
# 9. RANDOM FOREST PERFORMANCE
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

# ---------------------------------------------------------
# 10. BASELINE
# ---------------------------------------------------------

# Baseline:
# Predict tomorrow's return as 0%

baseline_pred = np.zeros(len(y_test))

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

# ---------------------------------------------------------
# 11. DISPLAY RESULTS
# ---------------------------------------------------------

print()
print("MODEL PERFORMANCE")
print("-" * 70)

print(f"Random Forest MAE:  {mae:.4f}%")
print(f"Random Forest RMSE: {rmse:.4f}%")
print(f"Random Forest R²:   {r2:.4f}")

print()
print("BASELINE PERFORMANCE")
print("-" * 70)

print(f"Baseline MAE:  {baseline_mae:.4f}%")
print(f"Baseline RMSE: {baseline_rmse:.4f}%")
print(f"Baseline R²:   {baseline_r2:.4f}")

# ---------------------------------------------------------
# 12. MODEL IMPROVEMENT
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

print(f"MAE improvement:  {mae_improvement:.2f}%")
print(f"RMSE improvement: {rmse_improvement:.2f}%")

# ---------------------------------------------------------
# 13. FEATURE IMPORTANCE
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
print("FEATURE IMPORTANCE")
print("-" * 70)

print(importance.to_string(index=False))

# ---------------------------------------------------------
# 14. CREATE PREDICTION DATASET
# ---------------------------------------------------------

predictions = test_data[
    ["Date", "Company"]
].copy()

predictions["Actual_Return"] = y_test.values

predictions["Predicted_Return"] = y_pred

predictions["Prediction_Error"] = (
    predictions["Actual_Return"]
    - predictions["Predicted_Return"]
)

# Direction prediction

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
# 15. DIRECTIONAL ACCURACY
# ---------------------------------------------------------

direction_accuracy = (
    predictions["Actual_Direction"]
    ==
    predictions["Predicted_Direction"]
).mean() * 100

print()
print("DIRECTION PREDICTION")
print("-" * 70)

print(
    f"Directional Accuracy: {direction_accuracy:.2f}%"
)

# ---------------------------------------------------------
# 16. SAVE RESULTS
# ---------------------------------------------------------

predictions.to_csv(
    "data/model_results/return_predictions.csv",
    index=False
)

importance.to_csv(
    "data/model_results/return_feature_importance.csv",
    index=False
)

comparison = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2 Score"
    ],
    "Baseline": [
        baseline_mae,
        baseline_rmse,
        baseline_r2
    ],
    "Random Forest": [
        mae,
        rmse,
        r2
    ]
})

comparison.to_csv(
    "data/model_results/return_model_comparison.csv",
    index=False
)

# ---------------------------------------------------------
# 17. SAMPLE PREDICTIONS
# ---------------------------------------------------------

print()
print("SAMPLE PREDICTIONS")
print("-" * 70)

print(
    predictions.head(10).to_string(index=False)
)

print()
print("Results saved to:")
print("data/model_results/return_predictions.csv")
print("data/model_results/return_feature_importance.csv")
print("data/model_results/return_model_comparison.csv")

print()
print("=" * 70)
print("RETURN PREDICTION MODEL COMPLETED")
print("=" * 70)