# -*- coding: utf-8 -*-
"""
Main training script for stock price forecasting models (SAMBA / LSTM)
Includes multi-stage evaluation (Train/Val/Test) and SHAP interpretability.
"""

import os
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from models.lstm import LSTM  
from utils import (
    prepare_data, init_seed, print_model_parameters,
    pearson_correlation, rank_information_coefficient, All_Metrics
)
from trainer import Trainer

# -------------------------
# HELPERS & LOSS FUNCTIONS
# -------------------------

def masked_mae_loss(scaler, mask_value):
    """Masked MAE loss function with automatic slicing for target matching"""
    def loss(preds, labels):
        # FIX: Ensure model predictions match label size to prevent broadcasting errors
        if preds.size(1) != labels.size(1):
            preds = preds[:, :labels.size(1), :]

        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        
        from utils.metrics import MAE_torch
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def calculate_metrics_for_dataset(trainer, loader, mmn):
    """Helper to calculate and scale metrics for a specific dataloader"""
    y1, y2 = trainer.test(trainer.model, trainer.args, loader, trainer.logger)
    
    # Scale back to original price levels (Target assumed at index 0)
    y_p = y1[:, 0, :] * (mmn.max[0] - mmn.min[0]) + mmn.min[0]
    y_t = y2[:, 0, :] * (mmn.max[0] - mmn.min[0]) + mmn.min[0]
    
    mae, rmse, _ = All_Metrics(y_p, y_t, None, None)
    ic = pearson_correlation(y_t, y_p)
    ric = rank_information_coefficient(y_t[:, 0], y_p[:, 0])
    
    return mae, rmse, ic, ric, y_p, y_t

# -------------------------
# SHAP ANALYSIS
# -------------------------

def run_shap_analysis(model, train_loader, test_loader, output_dir, file_prefix, feature_names=None):
    """
    Generates and saves a SHAP summary plot. 
    Aggregates dimensions independently to avoid "axis out of bounds" errors.
    """
    import shap
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")

    print("\n--- Running SHAP Analysis ---")
    model.eval()
    device = next(model.parameters()).device

    try:
        # 1. Extract Data
        batch = next(iter(test_loader))
        test_data = batch[0][:50].to(device)
        
        train_batch = next(iter(train_loader))
        background_data = train_batch[0][:100].to(device)

        # 2. Wrapper for single scalar output
        class TargetWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                out = self.m(x)
                if out.dim() >= 3:
                    return out[:, 0, 0].unsqueeze(1)
                return out.view(out.size(0), -1)[:, 0].unsqueeze(1)

        wrapper = TargetWrapper(model)

        # 3. Compute SHAP values with cuDNN safety
        print("Calculating gradients (this may take a moment)...")
        with torch.backends.cudnn.flags(enabled=False):
            explainer = shap.GradientExplainer(wrapper, background_data)
            shap_values = explainer.shap_values(test_data)
        
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        test_data_np = test_data.cpu().numpy()

        # 4. Dimensional Reduction (HANDLED INDEPENDENTLY)
        # Process SHAP values based on THEIR shape
        if sv.ndim == 4:
            sv_2d = sv.sum(axis=(1, 3))
        elif sv.ndim == 3:
            sv_2d = sv.sum(axis=1)
        else:
            sv_2d = sv

        # Process Input Data based on ITS shape (Fixes the axis out of bounds error)
        if test_data_np.ndim == 4:
            test_data_2d = test_data_np.mean(axis=(1, 3))
        elif test_data_np.ndim == 3:
            test_data_2d = test_data_np.mean(axis=1)
        else:
            test_data_2d = test_data_np

        # 5. Plotting
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv_2d, test_data_2d, feature_names=feature_names, show=False)
        
        plot_path = os.path.join(output_dir, f"{file_prefix}_shap_summary.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP summary plot saved to {plot_path}\n")

    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}\n")

# -------------------------
# MAIN EXECUTION
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="SAMBA/LSTM Training Script")
    parser.add_argument('--model', type=str, default='samba', choices=['samba', 'lstm'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='Dataset/sp500_with_indicators.csv')
    parser.add_argument('--num_features', type=int, default=None)
    cli_args = parser.parse_args()

    model_args, config = get_paper_config()
    dataset_info = get_dataset_info()

    seed = cli_args.seed if cli_args.seed is not None else config.seed
    dataset_file = cli_args.dataset
    model_choice = cli_args.model.lower()
    init_seed(seed)

    os.makedirs("results", exist_ok=True)
    csv_name = os.path.splitext(os.path.basename(dataset_file))[0]
    json_path = os.path.join("results", f"results_{model_choice}_{csv_name}_{seed}.json")
    txt_path = os.path.join("results", f"results_{model_choice}_{csv_name}_{seed}.txt")

    print(f"Preparing data: {dataset_file}")
    train_loader, val_loader, test_loader, mmn, data_num_features = prepare_data(
        csv_file=dataset_file, window=config.lag, predict=config.horizon,
        test_ratio=config.test_ratio, val_ratio=config.val_ratio
    )

    num_features = cli_args.num_features if cli_args.num_features is not None else data_num_features
    config.num_nodes = num_features
    args = config.to_dict()

    print(f"Initializing {model_choice.upper()}...")
    if model_choice == 'samba':
        model_args.vocab_size = num_features
        model = SAMBA(model_args, args.get('hid'), args.get('lag'), args.get('horizon'), args.get('embed_dim'), args.get("cheb_k"))
    else:
        model = LSTM(input_size=num_features, hidden_size=args.get('hid', 64), output_size=args.get('horizon'))
    
    model = model.cuda()
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.uniform_(p)

    # Optimization Setup with slicing fix
    if args.get('loss_func') == 'mask_mae':
        loss_fn = masked_mae_loss(mmn, mask_value=None)
    else:
        base_criterion = torch.nn.L1Loss() if args.get('loss_func') == 'mae' else torch.nn.MSELoss()
        base_criterion = base_criterion.to(args.get('device'))

        def loss_fn(preds, labels):
            if preds.size(1) != labels.size(1):
                preds = preds[:, :labels.size(1), :]
            return base_criterion(preds, labels)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.get('lr_init'))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(0.5*args.get('epochs'))], gamma=0.1) if args.get('lr_decay') else None

    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader, test_loader, args=args, lr_scheduler=lr_scheduler)

    print(f"Starting Training (Seed {seed})...")
    trainer.train()

    print("Gathering final metrics...")
    t_mae, t_rmse, t_ic, t_ric, _, _ = calculate_metrics_for_dataset(trainer, train_loader, mmn)
    v_mae, v_rmse, v_ic, v_ric, _, _ = calculate_metrics_for_dataset(trainer, val_loader, mmn)
    mae, rmse, ic, ric, y_p, y_t = calculate_metrics_for_dataset(trainer, test_loader, mmn)

    # Save JSON Results
    y_p_np, y_t_np = y_p.cpu().numpy(), y_t.cpu().numpy()
    pred_returns = (y_p_np[1:] - y_p_np[:-1]) / y_p_np[:-1]
    true_returns = (y_t_np[1:] - y_t_np[:-1]) / y_t_np[:-1]
    
    df_raw = pd.read_csv(dataset_file)
    num_pts = len(pred_returns)
    records = [{"date": d, "price": float(pr), "predicted_return": float(p[0]), "actual_return": float(t[0])} 
               for d, pr, p, t in zip(df_raw["Date"].iloc[-num_pts:].tolist(), df_raw["Price"].iloc[-num_pts:].tolist(), pred_returns.tolist(), true_returns.tolist())]

    with open(json_path, "w") as jf: json.dump(records, jf, indent=4)

    # Save Text Results
    with open(txt_path, "a") as f:
        f.write(f"--- MODEL: {model_choice.upper()} (Seed: {seed}) ---\n")
        f.write(f"TRAIN -> MAE: {t_mae:.4f}, RMSE: {t_rmse:.4f}, IC: {t_ic:.4f}, RIC: {t_ric:.4f}\n")
        f.write(f"VAL   -> MAE: {v_mae:.4f}, RMSE: {v_rmse:.4f}, IC: {v_ic:.4f}, RIC: {v_ric:.4f}\n")
        f.write(f"TEST  -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, IC: {ic:.4f}, RIC: {ric:.4f}\n\n")

    print(f"Final Test IC: {ic:.4f}, RIC: {ric:.4f}")

    # ---------------------------------------------------------
    # DYNAMIC SHAP NAMING (Adapts to any dataset size automatically!)
    # ---------------------------------------------------------
    # 1. Read the CSV and set 'Date' as the index, just like data_utils.py does
    df_raw = pd.read_csv(dataset_file, index_col="Date")

    # 2. Replicate the exact dropping logic from prepare_data
    if "Name" in df_raw.columns:
        del df_raw["Name"]
    if "Target" in df_raw.columns:
        del df_raw["Target"]

    # 3. Target_Return is dynamically generated and placed at the front
    base_cols = [col for col in df_raw.columns if col != "Target_Return"]
    feature_names = ["Target_Return"] + base_cols

    # 4. Truncate to the exact model input size just in case
    feature_names = feature_names[:num_features]

    # Safety Check
    if len(feature_names) != num_features:
        print(f"WARNING: Extracted {len(feature_names)} names but model expects {num_features}.")
        print("Falling back to generic names to prevent crash.")
        feature_names = [f"Feature {i}" for i in range(num_features)]
    else:
        print(f"DEBUG: Successfully mapped {len(feature_names)} dynamic names: {feature_names[:3]}...")

    run_shap_analysis(model, train_loader, test_loader, "results", f"results_{model_choice}_{csv_name}_{seed}", feature_names)
    print("Process complete.")

if __name__ == "__main__":
    main()