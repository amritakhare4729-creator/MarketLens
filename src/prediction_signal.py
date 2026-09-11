import pandas as pd
import numpy as np
import os

# ============================================================
# MARKETLENS - PREDICTION SIGNAL GENERATOR
# ============================================================

print("=" * 70)
print("MARKETLENS - PREDICTION SIGNAL GENERATOR")
print("=" * 70)


# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

input_file = "data/advanced_features.csv"

output_folder = "data/model_results"

output_file = os.path.join(
    output_folder,
    "market_signals.csv"
)


# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

os.makedirs(output_folder, exist_ok=True)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("\nLoading advanced feature dataset...")

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])

print(f"Rows loaded: {len(df)}")
print(f"Companies: {df['Company'].nunique()}")


# ------------------------------------------------------------
# Sort data
# ------------------------------------------------------------

df = df.sort_values(
    ["Company", "Date"]
).reset_index(drop=True)


# ------------------------------------------------------------
# Get latest record for every company
# ------------------------------------------------------------

latest = (
    df.groupby("Company")
    .tail(1)
    .copy()
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Create technical signal score
# ------------------------------------------------------------

latest["Signal_Score"] = 0


# ============================================================
# RULE 1 - Moving Average Trend
# ============================================================

latest.loc[
    latest["Close"] > latest["MA_20"],
    "Signal_Score"
] += 1

latest.loc[
    latest["Close"] < latest["MA_20"],
    "Signal_Score"
] -= 1


latest.loc[
    latest["Close"] > latest["MA_50"],
    "Signal_Score"
] += 1

latest.loc[
    latest["Close"] < latest["MA_50"],
    "Signal_Score"
] -= 1


# ============================================================
# RULE 2 - RSI
# ============================================================

# Oversold condition
latest.loc[
    latest["RSI_14"] < 30,
    "Signal_Score"
] += 1

# Overbought condition
latest.loc[
    latest["RSI_14"] > 70,
    "Signal_Score"
] -= 1


# ============================================================
# RULE 3 - MACD
# ============================================================

latest.loc[
    latest["MACD"] > latest["MACD_Signal"],
    "Signal_Score"
] += 1

latest.loc[
    latest["MACD"] < latest["MACD_Signal"],
    "Signal_Score"
] -= 1


# ============================================================
# RULE 4 - Recent Momentum
# ============================================================

latest.loc[
    latest["Return_5D"] > 0,
    "Signal_Score"
] += 1

latest.loc[
    latest["Return_5D"] < 0,
    "Signal_Score"
] -= 1


# ============================================================
# RULE 5 - Daily Return
# ============================================================

latest.loc[
    latest["Daily_Return"] > 0,
    "Signal_Score"
] += 1

latest.loc[
    latest["Daily_Return"] < 0,
    "Signal_Score"
] -= 1


# ============================================================
# Convert score into signal
# ============================================================

def generate_signal(score):

    if score >= 3:
        return "BUY / POSITIVE"

    elif score <= -3:
        return "SELL / NEGATIVE"

    else:
        return "HOLD / NEUTRAL"


latest["Technical_Signal"] = latest[
    "Signal_Score"
].apply(generate_signal)


# ------------------------------------------------------------
# Signal strength
# ------------------------------------------------------------

def signal_strength(score):

    if abs(score) >= 5:
        return "Strong"

    elif abs(score) >= 3:
        return "Moderate"

    else:
        return "Weak"


latest["Signal_Strength"] = latest[
    "Signal_Score"
].apply(signal_strength)


# ------------------------------------------------------------
# Trend classification
# ------------------------------------------------------------

def determine_trend(row):

    if (
        row["Close"] > row["MA_20"]
        and row["Close"] > row["MA_50"]
    ):
        return "Bullish"

    elif (
        row["Close"] < row["MA_20"]
        and row["Close"] < row["MA_50"]
    ):
        return "Bearish"

    else:
        return "Mixed"


latest["Trend"] = latest.apply(
    determine_trend,
    axis=1
)


# ------------------------------------------------------------
# Volatility classification
# ------------------------------------------------------------

volatility_median = latest[
    "Volatility_20"
].median()


def determine_volatility(value):

    if value >= volatility_median * 1.5:
        return "High"

    elif value <= volatility_median * 0.75:
        return "Low"

    else:
        return "Moderate"


latest["Volatility_Level"] = latest[
    "Volatility_20"
].apply(determine_volatility)


# ------------------------------------------------------------
# Prepare final output
# ------------------------------------------------------------

output_columns = [
    "Date",
    "Company",
    "Ticker",
    "Close",
    "Daily_Return",
    "Return_5D",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Volatility_20",
    "MA_20",
    "MA_50",
    "Signal_Score",
    "Technical_Signal",
    "Signal_Strength",
    "Trend",
    "Volatility_Level"
]


signals = latest[output_columns].copy()


# ------------------------------------------------------------
# Round numerical values
# ------------------------------------------------------------

numeric_columns = [
    "Close",
    "Daily_Return",
    "Return_5D",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Volatility_20",
    "MA_20",
    "MA_50"
]

signals[numeric_columns] = signals[
    numeric_columns
].round(4)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

signals.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MARKET SIGNALS")
print("=" * 70)

print(
    signals[
        [
            "Company",
            "Close",
            "RSI_14",
            "Trend",
            "Signal_Score",
            "Technical_Signal",
            "Signal_Strength",
            "Volatility_Level"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# Signal summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SIGNAL SUMMARY")
print("=" * 70)

print(
    signals["Technical_Signal"]
    .value_counts()
    .to_string()
)


print("\n" + "=" * 70)
print("TREND SUMMARY")
print("=" * 70)

print(
    signals["Trend"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SIGNAL GENERATION COMPLETE")
print("=" * 70)

print(f"\nSaved to:")
print(output_file)

print(
    "\nIMPORTANT:"
    "\nThese signals are technical decision-support indicators."
    "\nThey are NOT guaranteed stock recommendations or financial advice."
)