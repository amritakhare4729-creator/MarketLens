import pandas as pd
import os

# Folder containing individual stock datasets
data_folder = "data"

# Output file
output_file = "data/master_stock_data.csv"

# Get all CSV files except the old TCS file and the master file
files = [
    file for file in os.listdir(data_folder)
    if file.endswith(".csv")
    and file != "tcs_stock_data.csv"
    and file != "master_stock_data.csv"
]

print("=" * 60)
print("MARKETLENS - DATA CLEANING & COMBINING")
print("=" * 60)

print(f"\nStock files found: {len(files)}")

all_data = []

# Read each stock file
for file in sorted(files):

    filepath = os.path.join(data_folder, file)

    print(f"Processing: {file}")

    df = pd.read_csv(filepath)

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Sort by date
    df = df.sort_values("Date")

    # Add to list
    all_data.append(df)


# Combine all stocks
master_data = pd.concat(all_data, ignore_index=True)

# Sort by date and company
master_data = master_data.sort_values(
    ["Date", "Company"]
).reset_index(drop=True)

# Check for missing values
missing_values = master_data.isna().sum().sum()

# Check for duplicates
duplicate_rows = master_data.duplicated().sum()

# Save master dataset
master_data.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print(f"Total rows: {len(master_data)}")
print(f"Total columns: {len(master_data.columns)}")
print(f"Missing values: {missing_values}")
print(f"Duplicate rows: {duplicate_rows}")

print("\nColumns:")
print(list(master_data.columns))

print(f"\nMaster dataset saved to:")
print(output_file)

print("\nFirst 5 rows:")
print(master_data.head())