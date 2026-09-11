import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 70)
print("MARKETLENS - STOCK PRICE PREDICTION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "data/stock_features.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

print(f"Loaded dataset: {len(df)} rows")
print(f"Companies: {df['Company'].nunique()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")


# ============================================================
# 2. SORT DATA
# ============================================================

df = df.sort_values(["Company", "Date"]).reset_index(drop=True)


# ============================================================
# 3. CREATE TARGET
# ============================================================
# Target = NEXT TRADING DAY'S closing price

df["Target_Next_Close"] = df.groupby("Company")["Close"].shift(-1)


print("\nTarget created:")
print("Target_Next_Close = next trading day's Close")


# ============================================================
# 4. SELECT FEATURES
# ============================================================

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

X = df[features].copy()
y = df["Target_Next_Close"].copy()


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

model_data = pd.concat([X, y, df["Date"], df["Company"]], axis=1)

model_data = model_data.dropna().reset_index(drop=True)

print(f"\nRows available for modeling: {len(model_data)}")


# ============================================================
# 6. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================
# IMPORTANT:
# We do NOT randomly shuffle stock-market data.

split_date = model_data["Date"].quantile(0.80)

train_data = model_data[model_data["Date"] <= split_date]
test_data = model_data[model_data["Date"] > split_date]

X_train = train_data[features]
y_train = train_data["Target_Next_Close"]

X_test = test_data[features]
y_test = test_data["Target_Next_Close"]


print("\nTRAINING / TESTING SPLIT")
print("-" * 70)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")
print(f"Training until: {train_data['Date'].max().date()}")
print(f"Testing from:   {test_data['Date'].min().date()}")


# ============================================================
# 7. TRAIN RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

r2 = r2_score(y_test, predictions)


print("\nMODEL PERFORMANCE")
print("-" * 70)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


# ============================================================
# 10. CREATE PREDICTION DATASET
# ============================================================

results = test_data[["Date", "Company"]].copy()

results["Actual_Close"] = y_test.values
results["Predicted_Close"] = predictions

results["Prediction_Error"] = (
    results["Actual_Close"] - results["Predicted_Close"]
)

results["Error_Percentage"] = (
    abs(results["Prediction_Error"])
    / results["Actual_Close"]
) * 100


# ============================================================
# 11. SAVE PREDICTIONS
# ============================================================

os.makedirs("data/model_results", exist_ok=True)

prediction_file = "data/model_results/stock_predictions.csv"

results.to_csv(prediction_file, index=False)

print("\nPredictions saved to:")
print(prediction_file)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("-" * 70)

print(importance.to_string(index=False))


# ============================================================
# 13. SAVE FEATURE IMPORTANCE
# ============================================================

importance_file = "data/model_results/feature_importance.csv"

importance.to_csv(importance_file, index=False)

print("\nFeature importance saved to:")
print(importance_file)


# ============================================================
# 14. SAMPLE PREDICTIONS
# ============================================================

print("\nSAMPLE PREDICTIONS")
print("-" * 70)

print(
    results.head(10).to_string(index=False)
)


print("\n" + "=" * 70)
print("PREDICTIVE MODELING COMPLETED")
print("=" * 70)