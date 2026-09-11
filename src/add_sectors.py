import pandas as pd

# Load feature dataset
input_file = "data/stock_features.csv"
output_file = "data/stock_features.csv"

df = pd.read_csv(input_file)

# Sector mapping
sector_map = {
    "TCS": "IT",
    "Infosys": "IT",
    "HCLTech": "IT",
    "Wipro": "IT",

    "HDFC Bank": "Banking",
    "ICICI Bank": "Banking",
    "SBI": "Banking",
    "Axis Bank": "Banking",

    "M&M": "Automobile",
    "Maruti Suzuki": "Automobile",

    "ITC": "FMCG",
    "Hindustan Unilever": "FMCG",

    "Bharti Airtel": "Telecom",

    "Sun Pharma": "Pharma",
    "Dr Reddy's": "Pharma",

    "Reliance": "Energy",
    "ONGC": "Energy",
    "NTPC": "Energy",

    "Tata Steel": "Metals"
}

# Add sector column
df["Sector"] = df["Company"].map(sector_map)

# Check whether every company received a sector
missing_sectors = df[df["Sector"].isna()]["Company"].unique()

if len(missing_sectors) > 0:
    print("WARNING: These companies have no sector:")
    print(missing_sectors)
else:
    print("All companies successfully assigned a sector.")

# Save updated dataset
df.to_csv(output_file, index=False)

print("\nSector information added successfully!")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\nSector distribution:")
print(df[["Company", "Sector"]].drop_duplicates().sort_values("Sector"))