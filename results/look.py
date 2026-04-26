import os
import glob
import pandas as pd

def parse_metrics_line(line):
    """Extracts MAE, RMSE, IC, and RIC from a given output line."""
    # Expected line: "STAGE -> MAE: 0.0123, RMSE: 0.0345, IC: 0.1234, RIC: 0.0567"
    try:
        metrics_part = line.split("->")[1].strip()
        parts = metrics_part.split(",")
        
        metrics = {}
        for part in parts:
            key, value = part.split(":")
            metrics[key.strip()] = value.strip()
        return metrics
    except Exception:
        return {"MAE": "N/A", "RMSE": "N/A", "IC": "N/A", "RIC": "N/A"}

def analyze_results(results_dir="results"):
    # Find all .txt files
    search_pattern = os.path.join(results_dir, "*.txt")
    txt_files = sorted(glob.glob(search_pattern))

    if not txt_files:
        print(f"No .txt files found in the '{results_dir}' directory.")
        print("Checking the current directory instead...")
        txt_files = sorted(glob.glob("*.txt"))
        if not txt_files:
            print("No .txt files found. Make sure you have run the training scripts.")
            return

    all_data = []

    # Loop through and parse each file
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        
        # Extract Model, Dataset, and Seed from the filename
        # Expected format: results_{model}_{dataset_name}_{seed}.txt
        # Example: results_samba_sp500_with_indicators_llm_1.txt
        clean_name = filename.replace("results_", "").replace(".txt", "")
        parts = clean_name.split("_")
        
        try:
            model_name = parts[0].upper()
            seed = parts[-1]
            dataset_name = "_".join(parts[1:-1])
        except IndexError:
            model_name, dataset_name, seed = "Unknown", "Unknown", "Unknown"

        # Initialize default rows
        row_data = {
            "Model": model_name,
            "Dataset": dataset_name,
            "Seed": seed
        }

        # Parse the text file
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    if line.startswith("TRAIN ->"):
                        m = parse_metrics_line(line)
                        row_data.update({f"Train_{k}": v for k, v in m.items()})
                    elif line.startswith("VAL   ->"):
                        m = parse_metrics_line(line)
                        row_data.update({f"Val_{k}": v for k, v in m.items()})
                    elif line.startswith("TEST  ->"):
                        m = parse_metrics_line(line)
                        row_data.update({f"Test_{k}": v for k, v in m.items()})
        except Exception as e:
            print(f"Error parsing {filename}: {e}")

        all_data.append(row_data)

    # Convert to DataFrame for beautiful formatting and easy exporting
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Ensure columns are in a logical order (in case some files missed a line)
        cols = ["Model", "Dataset", "Seed", 
                "Train_MAE", "Train_RMSE", "Train_IC", "Train_RIC",
                "Val_MAE", "Val_RMSE", "Val_IC", "Val_RIC",
                "Test_MAE", "Test_RMSE", "Test_IC", "Test_RIC"]
        
        # Only keep columns that actually exist in the dataframe to prevent errors
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        # Print to terminal
        print("\n" + "=" * 120)
        print(" " * 45 + "EXPERIMENT RESULTS SUMMARY")
        print("=" * 120)
        
        # Set pandas display options so it doesn't wrap awkwardly in the terminal
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        print(df.to_string(index=False))
        print("=" * 120)
        
        # Save to CSV for the user's paper
        output_csv = "final_summary_metrics.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Successfully extracted {len(txt_files)} files.")
        print(f"✅ Data has been saved to '{output_csv}' so you can easily copy it into Excel or Word!")

if __name__ == "__main__":
    analyze_results()