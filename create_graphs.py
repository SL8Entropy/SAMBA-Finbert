import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Load the data
# Replace 'results.json' with your actual filename
csv = "results_sp500_with_indicators_llm_3"
with open(csv+'.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# --- STEP 1: Calculate Metrics ---

# Calculate Daily Absolute Error (MAE for that specific day)
df['daily_ae'] = np.abs(df['predicted_return'] - df['actual_return'])

# Calculate Volatility Proxy
# Since High/Low prices are missing for ATR, we use Rolling Standard Deviation (20-day)
# This is a standard measure for "Market Volatility"
df['volatility'] = df['actual_return'].rolling(window=20).std()

# Remove initial days where volatility is NaN
df_clean = df.dropna(subset=['volatility', 'daily_ae']).copy()

# --- STEP 2: Create Regimes (Bins) ---

# Define Percentiles
p25 = df_clean['volatility'].quantile(0.25)
p75 = df_clean['volatility'].quantile(0.75)
p95 = df_clean['volatility'].quantile(0.95)

# Function to assign bins
def assign_regime(vol):
    if vol <= p25:
        return 'Low Volatility'
    elif vol <= p75:
        return 'Normal Volatility'
    elif vol <= p95:
        return 'High Volatility'
    else:
        return 'Extreme/Crisis'

df_clean['regime'] = df_clean['volatility'].apply(assign_regime)

# --- STEP 3: Aggregate Data ---

# Calculate Mean MAE and Standard Deviation (for error bars) per regime
# We reindex to ensure the bars appear in the logical order (Low -> Extreme)
regime_order = ['Low Volatility', 'Normal Volatility', 'High Volatility', 'Extreme/Crisis']
regime_stats = df_clean.groupby('regime')['daily_ae'].agg(['mean', 'std']).reindex(regime_order)

# --- STEP 4: Plotting ---

plt.figure(figsize=(10, 6))

# Plot Bar Chart with Error Bars (yerr)
# capsize controls the width of the "whisker" caps
bars = plt.bar(regime_stats.index, 
               regime_stats['mean'], 
               yerr=regime_stats['std'], 
               capsize=10, 
               color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'], 
               alpha=0.8, 
               edgecolor='black')

# Add labels and title
plt.xlabel('Volatility Regime (20-Day Rolling Std Dev)', fontsize=12)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Optional: Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0001, round(yval, 5), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(csv+'.png')