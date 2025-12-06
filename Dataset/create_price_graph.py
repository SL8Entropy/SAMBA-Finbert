import pandas as pd
import matplotlib.pyplot as plt

# 1. Configuration: Map filenames to Chart Titles
files = {
    'combined_dataframe_DJI.csv': 'Dow Jones Industrial Average (DJI)',
    'combined_dataframe_NYSE.csv': 'NYSE Composite',
    'combined_dataframe_IXIC.csv': 'NASDAQ Composite (IXIC)'
}

# 2. Hard-set Market Events (Crashes/Downturns)
# You can add "Booms" here by defining the start/end dates and changing the color to green.
events = [
    {
        "name": "2011 Debt Crisis",
        "start": "2011-07-01", "end": "2011-10-01",
        "color": "red", "alpha": 0.15
    },
    {
        "name": "2015-16 Selloff",
        "start": "2015-08-01", "end": "2016-02-15",
        "color": "red", "alpha": 0.15
    },
    {
        "name": "2018 Q4 Slump",
        "start": "2018-10-01", "end": "2018-12-24",
        "color": "red", "alpha": 0.15
    },
    {
        "name": "Covid-19 Crash",
        "start": "2020-02-20", "end": "2020-04-07",
        "color": "red", "alpha": 0.3  # Darker highlight for emphasis
    },
    {
        "name": "2022 Bear Market",
        "start": "2022-01-03", "end": "2022-10-14",
        "color": "red", "alpha": 0.15
    }
]

def create_market_plot(filename, title, events):
    # Load the data
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    # Convert Date column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Initialize the Plot
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Price'], label='Price', color='#1f77b4', linewidth=1.5)

    # Loop through events and add highlighted regions
    for event in events:
        start_date = pd.to_datetime(event["start"])
        end_date = pd.to_datetime(event["end"])
        
        # Highlight the period
        plt.axvspan(start_date, end_date, color=event["color"], alpha=event["alpha"], label=event["name"])
        
        # Add a text label near the top of the chart for the event
        mid_date = start_date + (end_date - start_date) / 2
        plt.text(mid_date, df['Price'].max(), event['name'], 
                 rotation=90, verticalalignment='top', horizontalalignment='center', 
                 fontsize=9, color='darkred', fontweight='bold')

    # Formatting
    plt.title(f'{title} - Price History with Major Crashes', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Clean up legend (avoid duplicates if multiple events share labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')

    # Save and Show
    output_name = filename.replace('.csv', '_plot.png')
    plt.savefig(output_name)
    print(f"Graph saved as: {output_name}")
    plt.show()

# 3. Main execution loop
for filename, title in files.items():
    create_market_plot(filename, title, events)