import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# MARKETLENS - DATA VISUALIZATIONS
# ============================================================

print("=" * 70)
print("MARKETLENS - DATA VISUALIZATIONS")
print("=" * 70)

# Create output directory
output_dir = "data/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Load feature dataset
df = pd.read_csv("data/stock_features.csv")

df["Date"] = pd.to_datetime(df["Date"])

print(f"Loaded dataset: {len(df)} rows")
print(f"Companies: {df['Company'].nunique()}")
print()


# ============================================================
# 1. STOCK PERFORMANCE
# ============================================================

performance = (
    df.groupby("Company")
    .agg(
        Start_Price=("Close", "first"),
        End_Price=("Close", "last")
    )
    .reset_index()
)

performance["Return_%"] = (
    (performance["End_Price"] - performance["Start_Price"])
    / performance["Start_Price"]
) * 100

performance = performance.sort_values("Return_%")

plt.figure(figsize=(12, 8))

plt.barh(
    performance["Company"],
    performance["Return_%"]
)

plt.axvline(0, linewidth=1)

plt.title("MarketLens - 5-Year Stock Performance")
plt.xlabel("5-Year Return (%)")
plt.ylabel("Company")

plt.tight_layout()
plt.savefig(
    f"{output_dir}/stock_performance.png",
    dpi=300
)
plt.close()

print("1. Stock performance chart saved.")


# ============================================================
# 2. STOCK PRICE TRENDS
# ============================================================

# Normalize every stock to 100 at the beginning
df["Normalized_Price"] = (
    df["Close"]
    / df.groupby("Company")["Close"].transform("first")
) * 100

price_pivot = df.pivot(
    index="Date",
    columns="Company",
    values="Normalized_Price"
)

plt.figure(figsize=(14, 8))

for company in price_pivot.columns:
    plt.plot(
        price_pivot.index,
        price_pivot[company],
        label=company
    )

plt.title("MarketLens - Normalized Stock Price Trends")
plt.xlabel("Date")
plt.ylabel("Normalized Price (Starting Value = 100)")
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()
plt.savefig(
    f"{output_dir}/stock_price_trends.png",
    dpi=300
)
plt.close()

print("2. Stock price trends chart saved.")


# ============================================================
# 3. SECTOR PERFORMANCE
# ============================================================

sector_performance = (
    df.groupby("Sector")["Daily_Return"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.barh(
    sector_performance.index,
    sector_performance.values * 100
)

plt.axvline(0, linewidth=1)

plt.title("MarketLens - Average Daily Return by Sector")
plt.xlabel("Average Daily Return (%)")
plt.ylabel("Sector")

plt.tight_layout()
plt.savefig(
    f"{output_dir}/sector_performance.png",
    dpi=300
)
plt.close()

print("3. Sector performance chart saved.")


# ============================================================
# 4. VOLATILITY COMPARISON
# ============================================================

volatility = (
    df.groupby("Company")["Daily_Return"]
    .std()
    .sort_values()
)

plt.figure(figsize=(12, 8))

plt.barh(
    volatility.index,
    volatility.values * 100
)

plt.title("MarketLens - Stock Volatility Comparison")
plt.xlabel("Daily Volatility (%)")
plt.ylabel("Company")

plt.tight_layout()
plt.savefig(
    f"{output_dir}/volatility_comparison.png",
    dpi=300
)
plt.close()

print("4. Volatility comparison chart saved.")


# ============================================================
# 5. TRADING VOLUME
# ============================================================

volume = (
    df.groupby("Company")["Volume"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(12, 8))

plt.barh(
    volume.index,
    volume.values
)

plt.title("MarketLens - Average Trading Volume")
plt.xlabel("Average Trading Volume")
plt.ylabel("Company")

plt.tight_layout()
plt.savefig(
    f"{output_dir}/trading_volume.png",
    dpi=300
)
plt.close()

print("5. Trading volume chart saved.")


# ============================================================
# 6. MOVING AVERAGES - TCS
# ============================================================

tcs = df[df["Company"] == "TCS"].copy()
tcs = tcs.sort_values("Date")

plt.figure(figsize=(14, 7))

plt.plot(
    tcs["Date"],
    tcs["Close"],
    label="TCS Close Price"
)

plt.plot(
    tcs["Date"],
    tcs["MA_20"],
    label="20-Day Moving Average"
)

plt.plot(
    tcs["Date"],
    tcs["MA_50"],
    label="50-Day Moving Average"
)

plt.title("MarketLens - TCS Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

plt.tight_layout()
plt.savefig(
    f"{output_dir}/tcs_moving_averages.png",
    dpi=300
)
plt.close()

print("6. TCS moving average chart saved.")


# ============================================================
# 7. CORRELATION HEATMAP
# ============================================================

correlation = df.pivot(
    index="Date",
    columns="Company",
    values="Daily_Return"
).corr()

plt.figure(figsize=(12, 10))

plt.imshow(
    correlation,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90,
    fontsize=8
)

plt.yticks(
    range(len(correlation.index)),
    correlation.index,
    fontsize=8
)

plt.title("MarketLens - Stock Return Correlation")

plt.tight_layout()
plt.savefig(
    f"{output_dir}/correlation_heatmap.png",
    dpi=300
)
plt.close()

print("7. Correlation heatmap saved.")


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("ALL VISUALIZATIONS COMPLETED")
print("=" * 70)

print()
print("Charts saved inside:")
print("data/visualizations/")
print()

print("Files created:")
print("1. stock_performance.png")
print("2. stock_price_trends.png")
print("3. sector_performance.png")
print("4. volatility_comparison.png")
print("5. trading_volume.png")
print("6. tcs_moving_averages.png")
print("7. correlation_heatmap.png")