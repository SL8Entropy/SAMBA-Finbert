import pandas as pd
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# Extend pandas with the progress_apply method
tqdm.pandas() 
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MODEL_NAME = "ProsusAI/finbert"
INPUT_FILE = "sp500_headlines_2008_2024.csv"
OUTPUT_FILE = "daily_sentiment_clean.csv"

# Check for CUDA (NVIDIA GPU) or default to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Loads FinBERT and the tokenizer."""
    print(f"Loading FinBERT model ({DEVICE})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

def get_finbert_scores(text, tokenizer, model):
    """Calculates Pos, Neg, Neu probabilities and a Compound score."""
    if pd.isna(text) or not isinstance(text, str):
        return [0.0, 0.0, 0.0, 0.0]
    
    tokens = tokenizer.encode_plus(
        text, max_length=512, truncation=True, padding='max_length', return_tensors='pt'
    )
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    
    with torch.no_grad():
        outputs = model(**tokens)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    pos = probs[0][0].item()
    neg = probs[0][1].item()
    neu = probs[0][2].item()
    
    # Create a single numerical metric (-1.0 to 1.0)
    compound = pos - neg 
    
    return [pos, neg, neu, compound]

def process_time_series(df):
    """Applies after-hours and weekend shifts, groups by day, and imputes NaNs."""
    print("Aligning timestamps to market hours...")
    
    # 1. Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Time-Zone Shift: After 16:00 (4 PM) moves to the next day
    if df['Date'].dt.time.nunique() > 1:
        mask_after_hours = df['Date'].dt.hour >= 16
        df.loc[mask_after_hours, 'Date'] += pd.Timedelta(days=1)
        
    # 3. Weekend Shift: Move Saturday (5) to Monday (+2) and Sunday (6) to Monday (+1)
    # This handles both native weekend news AND Friday after-hours news that got shifted to Saturday
    df.loc[df['Date'].dt.dayofweek == 5, 'Date'] += pd.Timedelta(days=2)
    df.loc[df['Date'].dt.dayofweek == 6, 'Date'] += pd.Timedelta(days=1)

    # 4. Normalize to Date Only (Drop the time component)
    df['Date_Only'] = df['Date'].dt.normalize()

    print("Aggregating daily sentiment...")
    # 5. Average the scores for each day
    daily_sentiment = df.groupby('Date_Only')[['Compound_Sentiment', 'Positive_Score', 'Negative_Score', 'Neutral_Score']].mean()

    print("Imputing missing market days (Zero Leakage)...")
    # 6. Create a strict Business Day calendar (Monday-Friday only)
    bdate_range = pd.bdate_range(start=daily_sentiment.index.min(), end=daily_sentiment.index.max())
    
    # 7. Reindex to the business calendar and Fill NaNs with Neutral (0.0)
    # WARNING: Do not use interpolate() here to avoid looking into the future!
    daily_sentiment = daily_sentiment.reindex(bdate_range).fillna(0.0)
    
    # Format for saving
    final_df = daily_sentiment.reset_index()
    final_df.rename(columns={'index': 'Date'}, inplace=True)
    
    return final_df

def main():
    tokenizer, model = load_model()

    print(f"\n1. Loading raw data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("2. Extracting FinBERT Sentiment Scores (This may take a while)...")
    # Apply scoring function
    df[['Positive_Score', 'Negative_Score', 'Neutral_Score', 'Compound_Sentiment']] = \
        df['Title'].astype(str).progress_apply(
            lambda x: pd.Series(get_finbert_scores(x, tokenizer, model))
        )

    print("\n3. Processing Time-Series structure...")
    final_df = process_time_series(df)

    print(f"\n4. Saving final clean dataset to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("✅ Pipeline Complete! Data is ready for the SAMBA model.")

if __name__ == "__main__":
    main()