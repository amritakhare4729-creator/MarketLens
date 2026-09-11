import pandas as pd
import os

# Input and output files
input_file = "data/master_stock_data.csv"
output_file = "data/stock_features.csv"

# Load master dataset
df = pd.read_csv(input_file)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort data properly
df = df.sort_values(["Company", "Date"]).reset_index(drop=True)

# Daily return
df["Daily_Return"] = (
    df.groupby("Company")["Close"]
    .pct_change() * 100
)

# Price change
df["Price_Change"] = (
    df.groupby("Company")["Close"]
    .diff()
)

# 20-day moving average
df["MA_20"] = (
    df.groupby("Company")["Close"]
    .transform(lambda x: x.rolling(20).mean())
)

# 50-day moving average
df["MA_50"] = (
    df.groupby("Company")["Close"]
    .transform(lambda x: x.rolling(50).mean())
)

# 20-day rolling volatility
df["Volatility_20"] = (
    df.groupby("Company")["Daily_Return"]
    .transform(lambda x: x.rolling(20).std())
)

# High-Low price range
df["High_Low_Range"] = df["High"] - df["Low"]

# Percentage high-low range
df["High_Low_Range_Pct"] = (
    (df["High"] - df["Low"]) / df["Close"]
) * 100

# Cumulative return for each stock
df["Cumulative_Return"] = (
    df.groupby("Company")["Daily_Return"]
    .transform(lambda x: (1 + x / 100).cumprod() - 1)
) * 100

# Save feature dataset
df.to_csv(output_file, index=False)

print("=" * 60)
print("MARKETLENS - FEATURE ENGINEERING")
print("=" * 60)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nNew features created:")
print("1. Daily_Return")
print("2. Price_Change")
print("3. MA_20")
print("4. MA_50")
print("5. Volatility_20")
print("6. High_Low_Range")
print("7. High_Low_Range_Pct")
print("8. Cumulative_Return")

print("\nMissing values after feature engineering:")
print(df.isna().sum())

print(f"\nFeature dataset saved to:")
print(output_file)

print("\nSample data:")
print(df.head())