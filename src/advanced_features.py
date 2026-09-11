import pandas as pd
import numpy as np

print("=" * 70)
print("MARKETLENS - ADVANCED FEATURE ENGINEERING")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD EXISTING FEATURE DATA
# ---------------------------------------------------------

df = pd.read_csv("data/stock_features.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Company", "Date"]
).reset_index(drop=True)

print(f"Loaded dataset: {len(df)} rows")
print(f"Companies: {df['Company'].nunique()}")

# ---------------------------------------------------------
# 2. PREVIOUS-DAY RETURNS
# ---------------------------------------------------------

df["Return_1D"] = (
    df.groupby("Company")["Close"]
    .pct_change(1)
)

df["Return_3D"] = (
    df.groupby("Company")["Close"]
    .pct_change(3)
)

df["Return_5D"] = (
    df.groupby("Company")["Close"]
    .pct_change(5)
)

df["Return_10D"] = (
    df.groupby("Company")["Close"]
    .pct_change(10)
)

# ---------------------------------------------------------
# 3. MOMENTUM FEATURES
# ---------------------------------------------------------

df["Momentum_5D"] = (
    df.groupby("Company")["Close"]
    .diff(5)
)

df["Momentum_10D"] = (
    df.groupby("Company")["Close"]
    .diff(10)
)

# ---------------------------------------------------------
# 4. PRICE VS MOVING AVERAGE
# ---------------------------------------------------------

df["Price_vs_MA20"] = (
    (df["Close"] - df["MA_20"])
    / df["MA_20"]
) * 100

df["Price_vs_MA50"] = (
    (df["Close"] - df["MA_50"])
    / df["MA_50"]
) * 100

# ---------------------------------------------------------
# 5. VOLUME FEATURES
# ---------------------------------------------------------

df["Volume_Change"] = (
    df.groupby("Company")["Volume"]
    .pct_change()
)

df["Volume_MA20"] = (
    df.groupby("Company")["Volume"]
    .transform(
        lambda x: x.rolling(20).mean()
    )
)

df["Volume_Ratio"] = (
    df["Volume"] / df["Volume_MA20"]
)

# ---------------------------------------------------------
# 6. RSI - 14 DAY
# ---------------------------------------------------------

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


df["RSI_14"] = (
    df.groupby("Company")["Close"]
    .transform(calculate_rsi)
)

# ---------------------------------------------------------
# 7. MACD
# ---------------------------------------------------------

def calculate_macd(series):

    ema12 = series.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = series.ewm(
        span=26,
        adjust=False
    ).mean()

    return ema12 - ema26


df["MACD"] = (
    df.groupby("Company")["Close"]
    .transform(calculate_macd)
)

# ---------------------------------------------------------
# 8. MACD SIGNAL
# ---------------------------------------------------------

df["MACD_Signal"] = (
    df.groupby("Company")["MACD"]
    .transform(
        lambda x: x.ewm(
            span=9,
            adjust=False
        ).mean()
    )
)

df["MACD_Histogram"] = (
    df["MACD"] - df["MACD_Signal"]
)

# ---------------------------------------------------------
# 9. VOLATILITY FEATURES
# ---------------------------------------------------------

df["Volatility_5D"] = (
    df.groupby("Company")["Daily_Return"]
    .transform(
        lambda x: x.rolling(5).std()
    )
)

df["Volatility_10D"] = (
    df.groupby("Company")["Daily_Return"]
    .transform(
        lambda x: x.rolling(10).std()
    )
)

# ---------------------------------------------------------
# 10. PRICE RANGE FEATURES
# ---------------------------------------------------------

df["Open_Close_Change"] = (
    (df["Close"] - df["Open"])
    / df["Open"]
) * 100

df["High_Low_Pct"] = (
    (df["High"] - df["Low"])
    / df["Low"]
) * 100

# ---------------------------------------------------------
# 11. TARGET
# ---------------------------------------------------------

df["Next_Day_Return"] = (
    df.groupby("Company")["Close"]
    .shift(-1)
    / df["Close"]
    - 1
) * 100

# ---------------------------------------------------------
# 12. COUNT NEW FEATURES
# ---------------------------------------------------------

new_features = [
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

print()
print("NEW FEATURES CREATED")
print("-" * 70)

for feature in new_features:
    print(f"✓ {feature}")

# ---------------------------------------------------------
# 13. DISPLAY DATASET INFORMATION
# ---------------------------------------------------------

print()
print("DATASET INFORMATION")
print("-" * 70)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"New features: {len(new_features)}")

# ---------------------------------------------------------
# 14. MISSING VALUES
# ---------------------------------------------------------

print()
print("MISSING VALUES IN NEW FEATURES")
print("-" * 70)

for feature in new_features:
    missing = df[feature].isna().sum()
    print(f"{feature:<25} {missing}")

# ---------------------------------------------------------
# 15. SAVE DATASET
# ---------------------------------------------------------

output_path = "data/advanced_features.csv"

df.to_csv(
    output_path,
    index=False
)

print()
print("Advanced feature dataset saved to:")
print(output_path)

print()
print("=" * 70)
print("ADVANCED FEATURE ENGINEERING COMPLETED")
print("=" * 70)