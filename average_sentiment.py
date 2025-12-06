import pandas as pd
import numpy as np

def process_sentiment_data(input_file, output_file):
    # 1. Load the data
    df = pd.read_csv(input_file)
    
    # Ensure Date is in datetime format
    # Expecting format like "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD"
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Apply "Time-Zone Trap" Fix
    # Logic: If time is after 16:00 (4:00 PM), shift to the next day.
    
    # We check if the dataset actually has time components (not just 00:00:00)
    # If your data is just dates (2024-01-01), this step is skipped automatically.
    if df['Date'].dt.time.nunique() > 1:
        print("Time data detected. Applying Time-Zone shift logic...")
        
        # Define the cutoff time (16:00:00)
        cutoff_hour = 16
        
        # Identify rows where the hour is greater than or equal to 16 (4 PM)
        # Note: If news happens exactly at 4:00 PM, we usually count it as after market.
        # Adjust logic strictly > 16 or >= 16 based on preference. Here we use >= 16.
        mask = df['Date'].dt.hour >= cutoff_hour
        
        # Shift those dates by 1 day
        df.loc[mask, 'Date'] = df.loc[mask, 'Date'] + pd.Timedelta(days=1)
    else:
        print("No specific time information found (only dates). Skipping time shift.")

    # Normalize to Date only (remove time component for grouping)
    df['Date_Only'] = df['Date'].dt.normalize()

    # 3. Average sentiment each day
    daily_sentiment = df.groupby('Date_Only')['Sentiment'].mean()

    # 4. Interpolate missing days
    # Create a full date range from the start to the end of the dataset
    full_date_range = pd.date_range(start=daily_sentiment.index.min(), 
                                    end=daily_sentiment.index.max(), 
                                    freq='D')
    
    # Reindex the series to this full range (this creates NaNs for missing days)
    daily_sentiment = daily_sentiment.reindex(full_date_range)
    
    # Interpolate the missing values (linear interpolation handles gaps smoothly)
    daily_sentiment_interpolated = daily_sentiment.interpolate(method='linear')
    
    # 5. Save to CSV
    # Convert back to DataFrame for saving
    final_df = daily_sentiment_interpolated.reset_index()
    final_df.columns = ['Date', 'Sentiment']
    
    final_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

# Execution
if __name__ == "__main__":
    process_sentiment_data('Dataset/sp500_headlines_with_scores_formatted.csv', 'Dataset/daily_sentiment_interpolated.csv')