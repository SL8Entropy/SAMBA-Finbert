import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
from tqdm.auto import tqdm 

# Extend pandas with the progress_apply method
tqdm.pandas() 

# Suppress the UserWarning about to_json output in transformers
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MODEL_NAME = "ProsusAI/finbert"
INPUT_FILE = "Dataset/sp500_headlines_2008_2024.csv"
OUTPUT_FILE = "Dataset/sp500_headlines_with_scores.csv"

# --- Determine Device ---
# Check for CUDA (NVIDIA GPU) or default to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Load Model and Tokenizer ---
try:
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load model and move it to the selected device (GPU or CPU)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    
    print(f"Successfully loaded FinBERT model: {MODEL_NAME}")
    # Note on output order: The ProsusAI/finbert model typically outputs probabilities 
    # in the order: [Positive, Negative, Neutral].
except Exception as e:
    print(f"Error loading FinBERT model or tokenizer: {e}")
    print("Please ensure you have 'transformers', 'torch', and 'pandas' installed.")
    exit()

def get_finbert_scores(text):
    """
    Analyzes the sentiment of a given text using FinBERT and returns the 
    probabilities for Positive, Negative, and Neutral sentiment.
    """
    if pd.isna(text) or not isinstance(text, str):
        # Return zeros if the input is missing or invalid
        return [0.0, 0.0, 0.0]
    
    # Tokenize the input text
    tokens = tokenizer.encode_plus(
        text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Move tokens to the same device as the model
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    
    # Get model output (logits)
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Extract probabilities. FinBERT output order is: [Positive, Negative, Neutral]
    positive_score = probs[0][0].item()
    negative_score = probs[0][1].item()
    neutral_score = probs[0][2].item()
    
    # Return the three scores
    return [positive_score, negative_score, neutral_score]

# --- Main execution ---
def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Please ensure it is in the same directory.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'Title' not in df.columns:
        print("Error: DataFrame must contain a column named 'Title'.")
        print("Available columns:", df.columns.tolist())
        return

    print("Analyzing sentiment for each title and extracting scores...")
    
    # Apply the function which returns [P, N, Nu] scores as a Series, 
    # expanding the result into three new columns simultaneously.
    df[['Positive_Score', 'Negative_Score', 'Neutral_Score']] = \
        df['Title'].astype(str).progress_apply(
            lambda x: pd.Series(get_finbert_scores(x))
        )

    print(f"Saving results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Done! The results are saved in '{OUTPUT_FILE}' with three new score columns.")

if __name__ == "__main__":
    main()