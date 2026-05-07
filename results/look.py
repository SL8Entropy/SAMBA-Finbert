import os
import glob
import pandas as pd
import numpy as np

def parse_metrics_line(line):
    """Extracts MAE, RMSE, IC, and RIC from a given output line."""
    # Expected line: "STAGE -> MAE: 0.0123, RMSE: 0.0345, IC: 0.1234, RIC: 0.0567"
    try:
        metrics_part = line.split("->")[1].strip()
        parts = metrics_part.split(",")
        
        metrics = {}
        for part in parts:
            key, value = part.split(":")
            metrics[key.strip()] = float(value.strip()) # Cast to float for easier aggregation later
        return metrics
    except Exception:
        return {"MAE": np.nan, "RMSE": np.nan, "IC": np.nan, "RIC": np.nan}

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
        clean_name = filename.replace("results_", "").replace(".txt", "")
        parts = clean_name.split("_")
        
        try:
            model_name = parts[0].upper()
            seed = parts[-1]
            dataset_name = "_".join(parts[1:-1])
            
            # Make the dataset name prettier for the table
            if "llm" in dataset_name.lower():
                dataset_name = "S&P 500 (w/ Sentiment)"
            else:
                dataset_name = "S&P 500 (Base)"
                
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
        
        cols = ["Model", "Dataset", "Seed", 
                "Train_MAE", "Train_RMSE", "Train_IC", "Train_RIC",
                "Val_MAE", "Val_RMSE", "Val_IC", "Val_RIC",
                "Test_MAE", "Test_RMSE", "Test_IC", "Test_RIC"]
        
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        # ----------------------------------------------------
        # AGGREGATION LOGIC (Averaging across the 5 seeds)
        # ----------------------------------------------------
        # Group by Model and Dataset, then calculate the mean and standard deviation for the test metrics
        agg_df = df.groupby(["Model", "Dataset"]).agg({
            "Test_MAE": ['mean', 'std'],
            "Test_RMSE": ['mean', 'std'],
            "Test_IC": ['mean', 'std'],
            "Test_RIC": ['mean', 'std']
        }).reset_index()
        
        # Flatten the multi-level columns created by the aggregation
        agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

        # Formatting to 4 decimal places for the table
        format_mapping = {col: '{:.4f}' for col in agg_df.columns if 'mean' in col or 'std' in col}
        for col, fmt in format_mapping.items():
            agg_df[col] = agg_df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else "NaN")


        # Print Raw Runs to terminal
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        
        print("\n" + "=" * 120)
        print(" " * 45 + "RAW EXPERIMENT RESULTS (ALL SEEDS)")
        print("=" * 120)
        print(df.to_string(index=False))
        
        print("\n" + "=" * 120)
        print(" " * 40 + "AGGREGATED TEST RESULTS (MEAN ± STD)")
        print("=" * 120)
        print(agg_df.to_string(index=False))
        print("=" * 120)
        
        # Save to CSVs
        raw_csv = "final_raw_metrics.csv"
        agg_csv = "final_aggregated_metrics.csv"
        
        df.to_csv(raw_csv, index=False)
        agg_df.to_csv(agg_csv, index=False)
        
        print(f"\n✅ Successfully extracted {len(txt_files)} files.")
        print(f"✅ Raw Data saved to '{raw_csv}'")
        print(f"✅ Aggregated averages saved to '{agg_csv}'")

if __name__ == "__main__":
    analyze_results()