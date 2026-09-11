import pandas as pd
import os

# Folder containing our stock CSV files
data_folder = "data"

# Get all CSV files
files = [
    file for file in os.listdir(data_folder)
    if file.endswith(".csv")
]

print("=" * 60)
print("MARKETLENS - DATA VERIFICATION")
print("=" * 60)

print(f"\nTotal CSV files found: {len(files)}\n")

# Check every file
for file in sorted(files):

    filepath = os.path.join(data_folder, file)

    try:
        df = pd.read_csv(filepath)

        print(f"{file}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Missing values: {df.isna().sum().sum()}")
        print(f"  Duplicate rows: {df.duplicated().sum()}")
        print("-" * 60)

    except Exception as e:
        print(f"ERROR reading {file}: {e}")
        print("-" * 60)

print("\nVerification complete!")