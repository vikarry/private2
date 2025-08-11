import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Read the CSV file
df = pd.read_csv('your_file.csv')

# Convert month to datetime and set as index
df['month'] = pd.to_datetime(df['month'])
df.set_index('month', inplace=True)
df.sort_index(inplace=True)

# Perform classical decomposition - additive
decomposition_add = seasonal_decompose(df['rate'], model='additive', period=12)

# Perform classical decomposition - multiplicative
decomposition_mult = seasonal_decompose(df['rate'], model='multiplicative', period=12)

# Create comprehensive plots
fig, axes = plt.subplots(4, 2, figsize=(18, 16))

# Additive decomposition plots
axes[0, 0].plot(df.index, df['rate'], color='blue', linewidth=1.5)
axes[0, 0].set_title('Original Time Series - Additive', fontweight='bold')
axes[0, 0].set_ylabel('Rate')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(decomposition_add.trend.index, decomposition_add.trend, color='red', linewidth=2)
axes[1, 0].set_title('Trend Component - Additive', fontweight='bold')
axes[1, 0].set_ylabel('Trend')
axes[1, 0].grid(True, alpha=0.3)

axes[2, 0].plot(decomposition_add.seasonal.index, decomposition_add.seasonal, color='green', linewidth=1)
axes[2, 0].set_title('Seasonal Component - Additive', fontweight='bold')
axes[2, 0].set_ylabel('Seasonal')
axes[2, 0].grid(True, alpha=0.3)

axes[3, 0].plot(decomposition_add.resid.index, decomposition_add.resid, color='purple', linewidth=1, alpha=0.7)
axes[3, 0].set_title('Residual Component - Additive', fontweight='bold')
axes[3, 0].set_ylabel('Residual')
axes[3, 0].set_xlabel('Month')
axes[3, 0].grid(True, alpha=0.3)
axes[3, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Multiplicative decomposition plots
axes[0, 1].plot(df.index, df['rate'], color='blue', linewidth=1.5)
axes[0, 1].set_title('Original Time Series - Multiplicative', fontweight='bold')
axes[0, 1].set_ylabel('Rate')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(decomposition_mult.trend.index, decomposition_mult.trend, color='red', linewidth=2)
axes[1, 1].set_title('Trend Component - Multiplicative', fontweight='bold')
axes[1, 1].set_ylabel('Trend')
axes[1, 1].grid(True, alpha=0.3)

axes[2, 1].plot(decomposition_mult.seasonal.index, decomposition_mult.seasonal, color='green', linewidth=1)
axes[2, 1].set_title('Seasonal Component - Multiplicative', fontweight='bold')
axes[2, 1].set_ylabel('Seasonal')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)

axes[3, 1].plot(decomposition_mult.resid.index, decomposition_mult.resid, color='purple', linewidth=1, alpha=0.7)
axes[3, 1].set_title('Residual Component - Multiplicative', fontweight='bold')
axes[3, 1].set_ylabel('Residual')
axes[3, 1].set_xlabel('Month')
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Create seasonal pattern comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Monthly seasonal patterns
df['month_num'] = df.index.month
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Additive seasonal pattern
seasonal_add_monthly = decomposition_add.seasonal.groupby(decomposition_add.seasonal.index.month).mean()
axes[0, 0].bar(range(1, 13), seasonal_add_monthly.values, color='skyblue', alpha=0.7)
axes[0, 0].set_title('Additive Seasonal Pattern by Month')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Seasonal Component')
axes[0, 0].set_xticks(range(1, 13))
axes[0, 0].set_xticklabels(month_names, rotation=45)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Multiplicative seasonal pattern
seasonal_mult_monthly = decomposition_mult.seasonal.groupby(decomposition_mult.seasonal.index.month).mean()
axes[0, 1].bar(range(1, 13), seasonal_mult_monthly.values, color='lightcoral', alpha=0.7)
axes[0, 1].set_title('Multiplicative Seasonal Pattern by Month')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Seasonal Component')
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels(month_names, rotation=45)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5)

# Residual analysis
axes[1, 0].hist(decomposition_add.resid.dropna(), bins=30, alpha=0.7, color='purple', edgecolor='black')
axes[1, 0].set_title('Additive Residuals Distribution')
axes[1, 0].set_xlabel('Residual Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

axes[1, 1].hist(decomposition_mult.resid.dropna(), bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].set_title('Multiplicative Residuals Distribution')
axes[1, 1].set_xlabel('Residual Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].axvline(x=1, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Print analysis summary
print("CLASSICAL DECOMPOSITION ANALYSIS")
print("=" * 50)

print(f"\nData Period: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
print(f"Total Observations: {len(df)}")
print(f"Original Series - Mean: {df['rate'].mean():.4f}, Std: {df['rate'].std():.4f}")

print("\nADDITIVE DECOMPOSITION:")
print("-" * 25)
print(f"Trend - Mean: {decomposition_add.trend.mean():.4f}, Std: {decomposition_add.trend.std():.4f}")
print(f"Seasonal - Mean: {decomposition_add.seasonal.mean():.4f}, Std: {decomposition_add.seasonal.std():.4f}")
print(f"Residual - Mean: {decomposition_add.resid.mean():.4f}, Std: {decomposition_add.resid.std():.4f}")

print("\nMULTIPLICATIVE DECOMPOSITION:")
print("-" * 30)
print(f"Trend - Mean: {decomposition_mult.trend.mean():.4f}, Std: {decomposition_mult.trend.std():.4f}")
print(f"Seasonal - Mean: {decomposition_mult.seasonal.mean():.4f}, Std: {decomposition_mult.seasonal.std():.4f}")
print(f"Residual - Mean: {decomposition_mult.resid.mean():.4f}, Std: {decomposition_mult.resid.std():.4f}")

print("\nSEASONAL PATTERNS:")
print("-" * 18)
print("Month\t\tAdditive\tMultiplicative")
for i, month in enumerate(month_names, 1):
    add_val = seasonal_add_monthly.iloc[i-1]
    mult_val = seasonal_mult_monthly.iloc[i-1]
    print(f"{month}\t\t{add_val:8.4f}\t{mult_val:8.4f}")

# Model fit comparison
add_fitted = decomposition_add.trend + decomposition_add.seasonal
mult_fitted = decomposition_mult.trend * decomposition_mult.seasonal

add_mse = np.mean((df['rate'] - add_fitted)**2)
mult_mse = np.mean((df['rate'] - mult_fitted)**2)

print(f"\nMODEL FIT COMPARISON:")
print("-" * 21)
print(f"Additive MSE: {add_mse:.6f}")
print(f"Multiplicative MSE: {mult_mse:.6f}")
print(f"Better model: {'Additive' if add_mse < mult_mse else 'Multiplicative'}")