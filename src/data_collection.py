import yfinance as yf
import os

# Stock list
stocks = {
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HCLTech": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "M&M": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Reliance": "RELIANCE.NS",
    "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS",
    "Tata Steel": "TATASTEEL.NS"
}

# Create data folder
os.makedirs("data", exist_ok=True)

# Download data for each stock
for company, ticker in stocks.items():

    print(f"\nDownloading {company} ({ticker})...")

    data = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    # Check if data was downloaded
    if data.empty:
        print(f"WARNING: No data found for {company}")
        continue

    # Remove extra ticker level
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    # Reset date
    data.reset_index(inplace=True)

    # Add company and ticker information
    data["Company"] = company
    data["Ticker"] = ticker

    # Save individual CSV
    filename = company.replace(" ", "_").replace("&", "and").replace("'", "") + ".csv"
    filepath = os.path.join("data", filename)

    data.to_csv(filepath, index=False)

    print(f"Saved: {filepath} | Rows: {len(data)}")

print("\n===================================")
print("ALL STOCK DATA COLLECTION COMPLETE")
print("===================================")