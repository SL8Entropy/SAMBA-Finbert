# -*- coding: utf-8 -*-
"""
Main training script for SAMBA stock price forecasting model
"""

import os
import json
import pandas as pd

import torch
import torch.nn as nn
import numpy as np
from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from utils import (
    prepare_data, init_seed, print_model_parameters,
    pearson_correlation, rank_information_coefficient, All_Metrics
)
from trainer import Trainer


def masked_mae_loss(scaler, mask_value):
    """Masked MAE loss function"""
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        from utils.metrics import MAE_torch
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss


def main():
    """Main training function using paper configuration"""
    # Get paper configuration
    model_args, config = get_paper_config()
    dataset_info = get_dataset_info()

    print("🚀 SAMBA: A Graph-Mamba Approach for Stock Price Prediction")
    print(f"📚 Paper: {dataset_info['paper_title']}")
    print(f"🏛️  Conference: {dataset_info['conference']}")
    print(f"👥 Authors: {', '.join(dataset_info['authors'])}")
    print(f"📊 Expected Features: {dataset_info['total_features']}")
    print("=" * 70)

    # Initialize seed
    init_seed(config.seed)

    print("Loading and preparing data...")

    available_datasets = [ds['file'] for ds in dataset_info['datasets']]

    # Default dataset
    dataset_file = 'Dataset/combined_dataframe_NYSE_LLM.csv'

    # Extract CSV name & seed for output file naming
    csv_name = os.path.splitext(os.path.basename(dataset_file))[0]
    seed = config.seed
    json_output_path = f"results_{csv_name}_{seed}.json"
    txt_output_path = f"results_{csv_name}_{seed}.txt"

    # Check Dataset folder
    if not os.path.exists('Dataset'):
        print("❌ Dataset folder not found!")
        for ds in available_datasets:
            print(f"  - Dataset/{ds}")
        return

    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found!")
        for ds in available_datasets:
            fp = f"Dataset/{ds}"
            print(f"{'  ✅' if os.path.exists(fp) else '  ❌'} {fp}")
        return

    train_loader, val_loader, test_loader, mmn, num_features = prepare_data(
        csv_file=dataset_file,
        window=config.lag,
        predict=config.horizon,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio
    )

    config.num_nodes = num_features
    print(f"Number of features (graph nodes): {num_features}")

    args = config.to_dict()

    print("Initializing SAMBA model...")
    model_args.vocab_size = num_features

    model = SAMBA(
        model_args,
        args.get('hid'),
        args.get('lag'),
        args.get('horizon'),
        args.get('embed_dim'),
        args.get("cheb_k")
    )

    model = model.cuda()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    print_model_parameters(model, only_num=False)

    # Loss
    if args.get('loss_func') == 'mask_mae':
        loss = masked_mae_loss(mmn, mask_value=0.0)
    elif args.get('loss_func') == 'mae':
        loss = torch.nn.L1Loss().to(args.get('device'))
    elif args.get('loss_func') == 'mse':
        loss = torch.nn.MSELoss().to(args.get('device'))
    else:
        raise ValueError(f"Unknown loss function: {args.get('loss_func')}")

    # Optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.get('lr_init'),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False
    )

    # Learning rate scheduler
    lr_scheduler = None
    if args.get('lr_decay'):
        print('Applying learning rate decay.')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[0.5 * args.get('epochs'), 0.7 * args.get('epochs'), 0.9 * args.get('epochs')],
            gamma=0.1
        )

    trainer = Trainer(
        model, loss, optimizer,
        train_loader, val_loader, test_loader,
        args=args, lr_scheduler=lr_scheduler
    )

    print("Starting training...")
    y_pred, y_true = trainer.train()

    print("Evaluating on test set...")
    y1, y2 = trainer.test(trainer.model, trainer.args, test_loader, trainer.logger)

    y_p = np.array(y1[:, 0, :].cpu())
    y_t = np.array(y2[:, 0, :].cpu())

    y_p = mmn.inverse_transform(y_p)
    y_t = mmn.inverse_transform(y_t)

    df = pd.read_csv(dataset_file)

    lag = config.lag
    horizon = config.horizon

    pred_returns = (y_p[1:] - y_p[:-1]) / y_p[:-1]
    true_returns = (y_t[1:] - y_t[:-1]) / y_t[:-1]

    # 1. Determine how many data points we have predictions for
    num_points = len(pred_returns)
    
    # 2. Slice the Date and price columns from the end of the dataframe
    prediction_dates = df["Date"].iloc[-num_points:].tolist()
    
    # Ensure the column name matches your CSV (e.g., "price", "Price", "Close")
    # Using "price" as requested
    actual_prices = df["Price"].iloc[-num_points:].tolist()

    records = []
    # 3. Zip the prices into the loop
    for date, price, p_ret, t_ret in zip(prediction_dates, actual_prices, pred_returns.tolist(), true_returns.tolist()):
        records.append({
            "date": date,
            "price": float(price),  # Added raw price to JSON
            "predicted_return": float(p_ret[0]),
            "actual_return": float(t_ret[0])
        })

    # -------------------------
    # SAVE JSON WITH NEW NAME
    # -------------------------
    with open(json_output_path, "w") as json_file:
        json.dump(records, json_file, indent=4)

    print(f"Saved {len(records)} return predictions to {json_output_path}")

    # Metrics
    y_p = torch.tensor(y_p)
    y_t = torch.tensor(y_t)

    return_p = (y_p[1:] - y_p[:-1]) / y_p[:-1]
    return_t = (y_t[1:] - y_t[:-1]) / y_t[:-1]

    mae, rmse, _ = All_Metrics(return_p, return_t, None, None)
    IC = pearson_correlation(return_t, return_p)
    RIC = rank_information_coefficient(return_t[:, 0], return_p[:, 0])

    print("Final Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Information Coefficient (IC): {IC:.4f}")
    print(f"Rank Information Coefficient (RIC): {RIC:.4f}")

    # -------------------------
    # SAVE TXT WITH NEW NAME
    # -------------------------
    with open(txt_output_path, "a") as f:
        f.write(f"IC: {np.array(IC)}\n")
        f.write(f"RIC: {np.array(RIC)}\n")
        f.write(f"MAE: {np.array(mae)}\n")
        f.write(f"RMSE: {np.array(rmse)}\n\n")

    print(f"Saved metrics to {txt_output_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
