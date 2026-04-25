import pandas as pd
import numpy as np

df = pd.read_csv('sp500_index.csv')
# For demonstration, we'll assume 'df' is already loaded.

# Ensure Date is a datetime object and sort chronologically 
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. Moving Averages
df['SMA_20'] = df['Price'].rolling(window=20).mean()
df['SMA_50'] = df['Price'].rolling(window=50).mean()
df['EMA_20'] = df['Price'].ewm(span=20, adjust=False).mean()

# 3. Bollinger Bands (20-day, 2 Standard Deviations)
df['BB_std'] = df['Price'].rolling(window=20).std()
df['BB_upper'] = df['SMA_20'] + (df['BB_std'] * 2)
df['BB_lower'] = df['SMA_20'] - (df['BB_std'] * 2)

# 4. Relative Strength Index (RSI - 14-day)
# We use Wilder's Smoothing for accurate RSI calculation
delta = df['Price'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(com=14 - 1, adjust=False).mean()
avg_loss = loss.ewm(com=14 - 1, adjust=False).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# 5. MACD (Moving Average Convergence Divergence)
df['EMA_12'] = df['Price'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Price'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

# 6. Daily Returns & Momentum
df['Daily_Return'] = df['Price'].pct_change()
df['Momentum_10'] = df['Price'].diff(periods=10) # Price change over 10 days

# =====================================================================
# MACHINE LEARNING PREP: SHIFTING TO PREVENT LEAKAGE
# =====================================================================
# If your target variable is today's return or tomorrow's price, 
# your features MUST be shifted by 1 so the model only knows 
# what happened up until the close of the *previous* day.

indicator_columns = [
    'SMA_20', 'SMA_50', 'EMA_20', 'BB_upper', 'BB_lower', 
    'RSI_14', 'MACD', 'MACD_Signal', 'Daily_Return', 'Momentum_10'
]

for col in indicator_columns:
    df[f'{col}_lag1'] = df[col].shift(1)

# Drop rows with NaN values created by rolling windows and shifting
df_clean = df.dropna().reset_index(drop=True)

print(df_clean.head())
# Save the cleaned and calculated data to a new CSV file
df_clean.to_csv('sp500_with_indicators.csv', index=False)