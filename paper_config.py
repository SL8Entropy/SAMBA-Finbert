# -*- coding: utf-8 -*-
"""
Configuration matching the original SAMBA paper implementation
Based on: "Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction"
IEEE ICASSP 2025
"""

from config import ModelArgs, TrainingConfig


def get_paper_config():
    """Get configuration matching the original SAMBA paper"""
    
    # Model configuration as per the paper
    model_args = ModelArgs(
        d_model=32,           # Model dimension
        n_layer=3,            # Number of Mamba layers
        vocab_size=26,        
        seq_in=5,             # Input sequence length
        seq_out=1,            # Prediction horizon
        d_state=128,          # State dimension
        expand=2,             # Expansion factor
        dt_rank='auto',       # Auto-calculated
        d_conv=3,             # Convolution kernel size
        pad_vocab_size_multiple=8,
        conv_bias=True,
        bias=False
    )
    
    # Training configuration as per the paper
    training_config = TrainingConfig(
        dataset='STOCK_DATA',
        lag=5,                # Input sequence length
        horizon=1,            # Prediction horizon
        num_nodes=26,         
        val_ratio=0.15,       # 15% validation
        test_ratio=0.15,      # 15% test
        input_dim=1,
        output_dim=1,
        embed_dim=10,         # Embedding dimension
        rnn_units=128,        # RNN units
        num_layers=3,         # Number of layers
        cheb_k=3,             # Chebyshev polynomial order
        d_in=32,              # Input dimension
        hid=32,               # Hidden dimension
        batch_size=32,        # Batch size
        epochs=200,          # Training epochs
        lr_init=0.001,        # Initial learning rate
        lr_decay=True,        # Learning rate decay
        lr_decay_rate=0.5,    # Decay rate
        lr_decay_step=[30, 60, 90],  # Decay steps
        early_stop=True,      # Early stopping
        early_stop_patience=30,  # Patience
        grad_norm=False,      # Gradient clipping
        max_grad_norm=5,      # Max gradient norm
        loss_func='mae',      # Loss function
        mae_thresh=None,      # MAE threshold
        mape_thresh=0,        # MAPE threshold
        device='cuda:0',      # Device
        seed=5,               # Random seed
        debug=True,           # Debug mode
        log_step=20,          # Log step
        log_dir='./'          # Log directory
    )
    
    return model_args, training_config


def get_dataset_info():
    """Get information about the three datasets used in the paper"""
    return {
        'datasets': [
            {
                'name': 'S&P500',
                'file': 'sp500_with_indicators.csv',
                'description': 'Standard & Poor\'s 500 Index with calculated indicators',
                'period': '2015-03-06 to 2024-03-04',
                'features': 26
            },
            {
                'name': 'S&P500 with news sentiment',
                'file': 'sp500_with_indicators_llm.csv',
                'description': 'Standard & Poor\'s 500 Index with calculated indicators and Finbert derived news sentiment',
                'period': '2015-03-06 to 2024-03-04',
                'features': 30
            },
        ],
        'total_features': 30,
        'time_period': '2015-03-06 to 2024-03-04',
        'paper_title': 'SAMBA-FinBERT: Leveraging News Sentiment as a Feature for Financial Forecasting',
        'conference': '',
        'authors': ['Sudharshan Sambathkumar', 'K.Abirami']
    }


def print_paper_info():
    """Print information about the paper and datasets"""
    info = get_dataset_info()
    
    print("=" * 70)
    print(f"Paper: {info['paper_title']}")
    print(f"Conference: {info['conference']}")
    print(f"Authors: {', '.join(info['authors'])}")
    print(f"Time Period: {info['time_period']}")
    print(f"Total Features: {info['total_features']}")
    print()
    
    print("Datasets:")
    for dataset in info['datasets']:
        print(f"  • {dataset['name']} ({dataset['file']})")
        print(f"    - {dataset['description']}")
        print(f"    - {dataset['features']} features")
        print(f"    - Period: {dataset['period']}")
        print()
    
    print("=" * 70)


if __name__ == "__main__":
    print_paper_info()
