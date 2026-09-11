import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("MARKETLENS - MODEL EVALUATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD FEATURE DATA
# ---------------------------------------------------------

df = pd.read_csv("data/stock_features.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Company", "Date"]
).reset_index(drop=True)

print(f"Loaded dataset: {len(df)} rows")
print(f"Companies: {df['Company'].nunique()}")

# ---------------------------------------------------------
# 2. CREATE NEXT-DAY ACTUAL CLOSE
# ---------------------------------------------------------

df["Actual_Next_Close"] = (
    df.groupby("Company")["Close"].shift(-1)
)

# ---------------------------------------------------------
# 3. LOAD RANDOM FOREST PREDICTIONS
# ---------------------------------------------------------

predictions = pd.read_csv(
    "data/model_results/stock_predictions.csv"
)

predictions["Date"] = pd.to_datetime(
    predictions["Date"]
)

print(f"Prediction records: {len(predictions)}")

# ---------------------------------------------------------
# 4. CREATE BASELINE
# ---------------------------------------------------------

# Baseline assumption:
# Tomorrow's closing price = today's closing price

baseline = df[
    ["Date", "Company", "Close", "Actual_Next_Close"]
].copy()

baseline = baseline.rename(
    columns={
        "Close": "Baseline_Prediction",
        "Actual_Next_Close": "Actual_Close"
    }
)

baseline = baseline.dropna()

# ---------------------------------------------------------
# 5. SELECT ONLY NECESSARY RF COLUMNS
# ---------------------------------------------------------

rf_predictions = predictions[
    ["Date", "Company", "Predicted_Close"]
].copy()

# ---------------------------------------------------------
# 6. MERGE BASELINE + RANDOM FOREST
# ---------------------------------------------------------

results = baseline.merge(
    rf_predictions,
    on=["Date", "Company"],
    how="inner"
)

print(f"Matched records: {len(results)}")

# ---------------------------------------------------------
# 7. GET ACTUAL AND PREDICTED VALUES
# ---------------------------------------------------------

actual = results["Actual_Close"]

baseline_pred = results["Baseline_Prediction"]

random_forest_pred = results["Predicted_Close"]

# ---------------------------------------------------------
# 8. RANDOM FOREST PERFORMANCE
# ---------------------------------------------------------

rf_mae = mean_absolute_error(
    actual,
    random_forest_pred
)

rf_rmse = np.sqrt(
    mean_squared_error(
        actual,
        random_forest_pred
    )
)

rf_r2 = r2_score(
    actual,
    random_forest_pred
)

# ---------------------------------------------------------
# 9. BASELINE PERFORMANCE
# ---------------------------------------------------------

baseline_mae = mean_absolute_error(
    actual,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        actual,
        baseline_pred
    )
)

baseline_r2 = r2_score(
    actual,
    baseline_pred
)

# ---------------------------------------------------------
# 10. DISPLAY MODEL COMPARISON
# ---------------------------------------------------------

print()
print("MODEL COMPARISON")
print("-" * 70)

print(
    f"{'Metric':<15}"
    f"{'Baseline':>20}"
    f"{'Random Forest':>20}"
)

print("-" * 55)

print(
    f"{'MAE':<15}"
    f"{baseline_mae:>20.4f}"
    f"{rf_mae:>20.4f}"
)

print(
    f"{'RMSE':<15}"
    f"{baseline_rmse:>20.4f}"
    f"{rf_rmse:>20.4f}"
)

print(
    f"{'R² Score':<15}"
    f"{baseline_r2:>20.4f}"
    f"{rf_r2:>20.4f}"
)

# ---------------------------------------------------------
# 11. CALCULATE IMPROVEMENT
# ---------------------------------------------------------

mae_improvement = (
    (baseline_mae - rf_mae)
    / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - rf_rmse)
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
# 12. CONCLUSION
# ---------------------------------------------------------

print()
print("CONCLUSION")
print("-" * 70)

if rf_mae < baseline_mae:
    print("Random Forest has lower MAE than the baseline.")
else:
    print("Baseline has lower MAE than Random Forest.")

if rf_rmse < baseline_rmse:
    print("Random Forest has lower RMSE than the baseline.")
else:
    print("Baseline has lower RMSE than Random Forest.")

if rf_r2 > baseline_r2:
    print("Random Forest has a higher R² score.")
else:
    print("Baseline has a higher R² score.")

# ---------------------------------------------------------
# 13. SAVE COMPARISON
# ---------------------------------------------------------

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
        rf_mae,
        rf_rmse,
        rf_r2
    ]
})

comparison.to_csv(
    "data/model_results/model_comparison.csv",
    index=False
)

print()
print("Comparison saved to:")
print("data/model_results/model_comparison.csv")

print()
print("=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)