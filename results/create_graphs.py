import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Find all JSON files in the current folder
json_files = glob.glob('*.json')

if not json_files:
    print("No JSON files found in the current directory.")

# Loop through every JSON file found
for file_path in json_files:
    # Extract the base filename without the '.json' extension
    base_name = os.path.splitext(file_path)[0]
    
    print(f"Processing {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)

        # Check if the required columns are in the file to avoid crashing on irrelevant JSONs
        if 'predicted_return' not in df.columns or 'actual_return' not in df.columns:
            print(f"  -> Skipping {file_path}: Missing 'predicted_return' or 'actual_return'.")
            continue

        # --- STEP 1: Calculate Metrics ---

        # Calculate Daily Absolute Error (MAE for that specific day)
        df['daily_ae'] = np.abs(df['predicted_return'] - df['actual_return'])

        # Calculate Volatility Proxy
        # Since High/Low prices are missing for ATR, we use Rolling Standard Deviation (20-day)
        df['volatility'] = df['actual_return'].rolling(window=20).std()

        # Remove initial days where volatility is NaN
        df_clean = df.dropna(subset=['volatility', 'daily_ae']).copy()
        
        if df_clean.empty:
            print(f"  -> Skipping {file_path}: Not enough data for rolling volatility.")
            continue

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
        bars = plt.bar(regime_stats.index, 
                       regime_stats['mean'], 
                       yerr=regime_stats['std'], 
                       capsize=10, 
                       color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'], 
                       alpha=0.8, 
                       edgecolor='black')

        # Add labels and title
        plt.title(f'Volatility Regimes vs MAE\n({base_name})', fontsize=14)
        plt.xlabel('Volatility Regime (20-Day Rolling Std Dev)', fontsize=12)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            if not np.isnan(yval):  # Handle cases where a bin might be completely empty
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0001, round(yval, 5), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        
        # Save figure - this inherently overwrites existing files with the same name
        output_image = f"{base_name}.png"
        plt.savefig(output_image)
        
        # CRITICAL: Close the plot so the next file's graph doesn't overlay on top of this one
        plt.close()
        
        print(f"  -> Successfully saved {output_image}")

    except Exception as e:
        print(f"  -> Error processing {file_path}: {e}")

print("All files processed!")