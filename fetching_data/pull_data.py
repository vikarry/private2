import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime

fred = Fred(api_key='fd3b894e3ac47e2419d09865ec11fa78')

# Define date range
start_date = '2000-01-01'
end_date = '2025-01-01'

all_data = {}

print("Fetching Treasury Yields...")
# Treasury Yields from FRED
treasury_tickers = {
    'DGS2': '2Y_Treasury',
    'DGS5': '5Y_Treasury',
    'DGS10': '10Y_Treasury',
    'DGS30': '30Y_Treasury',
    'DGS1': '1Y_Treasury',
    'DGS3': '3Y_Treasury',
    'DGS7': '7Y_Treasury'
}

for fred_ticker, name in treasury_tickers.items():
    try:
        data = fred.get_series(fred_ticker, start_date, end_date)
        all_data[name] = data
        print(f"  ✓ {name}")
    except:
        print(f"  ✗ Failed to fetch {name}")

print("\nFetching Economic Indicators...")
economic_indicators = {
    'CPIAUCSL': 'CPI',
    'UNRATE': 'Unemployment_Rate',
    'PAYEMS': 'NFP_Total',
    'GDP': 'GDP',
    'DFF': 'Fed_Funds_Rate',
    'CPILFESL': 'Core_CPI',
}

for fred_ticker, name in economic_indicators.items():
    try:
        data = fred.get_series(fred_ticker, start_date, end_date)
        all_data[name] = data
        print(f"  ✓ {name}")
    except:
        print(f"  ✗ Failed to fetch {name}")

print("\nFetching Currency Pairs...")
# Currency Pairs from Yahoo Finance
currency_pairs = {
    'EURUSD=X': 'EUR_USD',
    'GBPUSD=X': 'GBP_USD',
    'USDJPY=X': 'USD_JPY',
    'USDCHF=X': 'USD_CHF',
}

for ticker, name in currency_pairs.items():
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
        # Convert to Series to match FRED data format
        data = pd.Series(data)
        all_data[name] = data
        print(f"  ✓ {name}")
    except:
        print(f"  ✗ Failed to fetch {name}")

print("\nFetching Market Indices...")
# Market Indices from Yahoo Finance
indices = {
    '^VIX': 'VIX',
}

for ticker, name in indices.items():
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
        # Convert to Series to match FRED data format
        data = pd.Series(data)
        all_data[name] = data
        print(f"  ✓ {name}")
    except:
        print(f"  ✗ Failed to fetch {name}")

print("\nFetching Interest Rate Spreads...")
# Calculate some common spreads
try:
    # 10Y-2Y Spread
    if '10Y_Treasury' in all_data and '2Y_Treasury' in all_data:
        spread_10y2y = all_data['10Y_Treasury'] - all_data['2Y_Treasury']
        all_data['Spread_10Y_2Y'] = spread_10y2y
        print("  ✓ 10Y-2Y Spread")

    # 10Y-3M Spread (need to fetch 3M Treasury Bill)
    tbill_3m = fred.get_series('DGS3MO', start_date, end_date)
    if tbill_3m is not None and '10Y_Treasury' in all_data:
        spread_10y3m = all_data['10Y_Treasury'] - tbill_3m
        all_data['Spread_10Y_3M'] = spread_10y3m
        print("  ✓ 10Y-3M Spread")
except:
    print("  ✗ Failed to calculate some spreads")

print("\nCreating DataFrame...")
# Convert dict of Series to DataFrame
df = pd.DataFrame.from_dict(all_data, orient='columns')

print("\nProcessing for working days only...")
# Create a business day index (excludes weekends)
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# Define US business days (excluding weekends and US federal holidays)
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# Get all business days in our date range
all_business_days = pd.date_range(start=start_date, end=end_date, freq=us_bd)

# Reindex to business days only
df = df.reindex(all_business_days)

# Forward fill all data (this handles monthly/quarterly data appropriately)
df = df.fillna(method='ffill')

# Drop any remaining NaN rows at the beginning
df = df.dropna(how='all')

print(f"✓ Filtered to {len(df)} working days")
print(f"✓ Forward-filled monthly/quarterly data")

print("\nSaving data...")
df.to_csv('financial_data_2000_2025.csv')
print(f"✓ Data saved to 'financial_data_2000_2025.csv'")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df.index.min()} to {df.index.max()}")
print(f"  Columns: {list(df.columns)}")

df.to_pickle('financial_data_2000_2025.pkl')
print(f"✓ Data also saved to 'financial_data_2000_2025.pkl'")

print("\nDataFrame Info:")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB")

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nData availability summary:")
print(df.count())