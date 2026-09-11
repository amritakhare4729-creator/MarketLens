import pandas as pd

# Load dataset
file = "data/stock_features.csv"
df = pd.read_csv(file)

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

print("=" * 70)
print("MARKETLENS - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# --------------------------------------------------
# 1. BASIC INFORMATION
# --------------------------------------------------

print("\n1. DATASET INFORMATION")
print("-" * 70)

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Number of companies: {df['Company'].nunique()}")
print(f"Number of sectors: {df['Sector'].nunique()}")
print(f"Start date: {df['Date'].min().date()}")
print(f"End date: {df['Date'].max().date()}")


# --------------------------------------------------
# 2. STOCK PERFORMANCE
# --------------------------------------------------

print("\n2. STOCK PERFORMANCE")
print("-" * 70)

performance = (
    df.groupby("Company")
    .agg(
        Start_Price=("Close", "first"),
        End_Price=("Close", "last"),
        Average_Price=("Close", "mean"),
        Average_Daily_Return=("Daily_Return", "mean"),
        Volatility=("Daily_Return", "std"),
        Average_Volume=("Volume", "mean")
    )
    .reset_index()
)

# Calculate total return
performance["Total_Return_%"] = (
    (performance["End_Price"] - performance["Start_Price"])
    / performance["Start_Price"]
) * 100

performance = performance.sort_values(
    "Total_Return_%",
    ascending=False
)

print("\nStock performance:")
print(
    performance[
        [
            "Company",
            "Start_Price",
            "End_Price",
            "Total_Return_%",
            "Volatility",
            "Average_Volume"
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 3. BEST AND WORST STOCK
# --------------------------------------------------

print("\n3. BEST AND WORST PERFORMERS")
print("-" * 70)

best_stock = performance.iloc[0]
worst_stock = performance.iloc[-1]

print(
    f"Best performer: {best_stock['Company']} "
    f"({best_stock['Total_Return_%']:.2f}%)"
)

print(
    f"Worst performer: {worst_stock['Company']} "
    f"({worst_stock['Total_Return_%']:.2f}%)"
)


# --------------------------------------------------
# 4. VOLATILITY
# --------------------------------------------------

print("\n4. STOCK VOLATILITY")
print("-" * 70)

volatility = performance.sort_values(
    "Volatility",
    ascending=False
)

print(
    volatility[
        ["Company", "Volatility"]
    ].to_string(index=False)
)


# --------------------------------------------------
# 5. SECTOR PERFORMANCE
# --------------------------------------------------

print("\n5. SECTOR PERFORMANCE")
print("-" * 70)

sector_performance = (
    df.groupby("Sector")
    .agg(
        Average_Return=("Daily_Return", "mean"),
        Average_Volatility=("Daily_Return", "std"),
        Average_Volume=("Volume", "mean")
    )
    .reset_index()
)

sector_performance = sector_performance.sort_values(
    "Average_Return",
    ascending=False
)

print(
    sector_performance.to_string(index=False)
)


# --------------------------------------------------
# 6. TRADING VOLUME
# --------------------------------------------------

print("\n6. HIGHEST AVERAGE TRADING VOLUME")
print("-" * 70)

volume = performance.sort_values(
    "Average_Volume",
    ascending=False
)

print(
    volume[
        ["Company", "Average_Volume"]
    ].to_string(index=False)
)


# --------------------------------------------------
# 7. SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)