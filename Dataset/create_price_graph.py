import pandas as pd
import matplotlib.pyplot as plt

# 1. Configuration
files = {'sp500_index.csv': 'S&P 500'}

# Date range for the study
START_STUDY = "2014-12-22"
END_STUDY = "2024-04-15"

# 2. Market Events (Crashes)
events = [
    {"name": "2015-16 Selloff", "start": "2015-08-01", "end": "2016-02-15", "color": "red", "alpha": 0.2},
    {"name": "2018 Q4 Slump", "start": "2018-10-01", "end": "2018-12-24", "color": "red", "alpha": 0.2},
    {"name": "Covid-19 Crash", "start": "2020-02-20", "end": "2020-04-07", "color": "red", "alpha": 0.4},
    {"name": "2022 Bear Market", "start": "2022-01-03", "end": "2022-10-14", "color": "red", "alpha": 0.2}
]

def create_market_plot(filename, title, events):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Filter data to your study period
    df = df[(df['Date'] >= START_STUDY) & (df['Date'] <= END_STUDY)].reset_index(drop=True)

    # Calculate Splits (70%, 15%, 15%)
    n = len(df)
    train_end_idx = int(n * 0.70)
    val_end_idx = int(n * 0.85)

    # Get Date markers for splits
    train_start = df['Date'].iloc[0]
    val_start = df['Date'].iloc[train_end_idx]
    test_start = df['Date'].iloc[val_end_idx]
    study_end = df['Date'].iloc[-1]

    # Initialize Plot
    plt.figure(figsize=(15, 8))
    plt.plot(df['Date'], df['Price'], label='S&P 500 Price', color='black', linewidth=1.2, zorder=5)

    # --- ADD DATA SPLIT HIGHLIGHTS ---
    # Training (70%) - Light Blue
    plt.axvspan(train_start, val_start, color='#e1f5fe', alpha=0.8, label='Training (70%)', zorder=1)
    # Validation (15%) - Light Orange
    plt.axvspan(val_start, test_start, color='#fff3e0', alpha=0.8, label='Validation (15%)', zorder=1)
    # Testing (15%) - Light Purple
    plt.axvspan(test_start, study_end, color='#f3e5f5', alpha=0.8, label='OOS Testing (15%)', zorder=1)

    # --- ADD MARKET EVENT HIGHLIGHTS ---
    for event in events:
        s_date = pd.to_datetime(event["start"])
        e_date = pd.to_datetime(event["end"])
        
        # Only plot events that fall within our filtered range
        if s_date >= df['Date'].min() and s_date <= df['Date'].max():
            plt.axvspan(s_date, e_date, color=event["color"], alpha=event["alpha"], zorder=2)
            
            # Label events at the top
            mid_date = s_date + (e_date - s_date) / 2
            plt.text(mid_date, df['Price'].max() * 0.98, event['name'], 
                     rotation=90, va='top', ha='center', fontsize=8, color='darkred', fontweight='bold', zorder=10)

    # Add descriptive labels for the periods at the bottom
    y_min = df['Price'].min()
    plt.text(train_start + (val_start-train_start)/2, y_min, "TRAIN", ha='center', fontweight='bold', alpha=0.5)
    plt.text(val_start + (test_start-val_start)/2, y_min, "VAL", ha='center', fontweight='bold', alpha=0.5)
    plt.text(test_start + (study_end-test_start)/2, y_min, "TEST", ha='center', fontweight='bold', alpha=0.5)

    # Formatting
    plt.title(f'{title} - Data Splits and Market Events ({START_STUDY} to {END_STUDY})', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.4)
    
    # Clean Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=True, facecolor='white')

    plt.tight_layout()
    output_name = filename.replace('.csv', '_study_split_plot.png')
    plt.savefig(output_name, dpi=300)
    print(f"Graph saved as: {output_name}")
    plt.show()

for filename, title in files.items():
    create_market_plot(filename, title, events)