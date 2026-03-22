# SAMBA-FinBERT: Leveraging News Sentiment as a Feature for S&P 500 Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **A PyTorch implementation of a hybrid forecasting architecture that integrates SAMBA (Mamba-based State Space Model) with FinBERT sentiment analysis.**

---

## 📌 Overview

This repository implements **SAMBA-FinBERT**, a model designed to tackle non-stationary financial time series forecasting.

It addresses the limitations of standard architectures (LSTMs/Transformers) by combining:

1.  **SAMBA Backbone:** An adaptation of the Mamba architecture for linear-time sequence modeling of quantitative data (OHLCV).
2.  **FinBERT Integration:** A pre-trained financial LLM to process news headlines and inject "qualitative" sentiment signals into the predictive loop.
3.  **Walk-Forward Validation:** A rigorous testing protocol that eliminates look-ahead bias, ensuring the model is evaluated under realistic "live trading" conditions.

---

## 🏗️ System Architecture

The model processes two distinct data streams:

* **Quantitative Stream:** Historical price data (Open, High, Low, Close, Volume) is processed via the **SAMBA** block, which utilizes a Selective Scan Mechanism (S6) to capture long-term dependencies without the quadratic computational cost of attention.
* **Qualitative Stream:** News headlines are processed by **FinBERT** to generate a bounded daily sentiment score $[-1, 1]$. This score serves as a dynamic covariate, modulating the model's state transitions during high-volatility periods.

---

## 🚀 Installation

### Prerequisites

* Python 3.9+
* CUDA-enabled GPU (Highly recommended for Mamba training)

### 1. Clone the Repository

```bash
git clone [https://github.com/SL8Entropy/SAMBA-Finbert.git](https://github.com/SL8Entropy/SAMBA-Finbert.git)
cd SAMBA-Finbert
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust version as needed)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install Mamba and other requirements
pip install packaging
pip install mamba-ssm causal-conv1d
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Data Preparation

This project requires two datasets:

* **News Data:** S&P 500 News Headlines (or similar financial news corpus).
* **Price Data:** OHLCV data for target indices (e.g., NYSE, NASDAQ, DJIA).

Run the preprocessing script to tokenize news and generate daily sentiment scores:

```bash
python scripts/preprocess_finbert.py --input data/raw_news.csv --output data/sentiment_scores.csv
```

### 2. Training

To train the model using the Walk-Forward protocol:

```bash
python train.py \
    --model samba_finbert \
    --index NYSE \
    --epochs 100 \
    --batch_size 64 \
    --seed 1
```

**Arguments:**

* `--model`: Choose between `samba` (baseline) or `samba_finbert` (hybrid).
* `--index`: The target market index (e.g., NYSE, IXIC, DJI).
* `--window_size`: The lookback window for the time series (default: 60 days).

### 3. Evaluation

To run the evaluation script and generate metrics (IC, RIC, RMSE, MAE):

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --test_file data/test_set.csv
```

---

## 📊 Performance Metrics

The implementation tracks four key metrics:

* **IC (Information Coefficient):** Pearson correlation between predicted and actual returns.
* **RIC (Rank IC):** Spearman rank correlation.
* **MAE:** Mean Absolute Error.
* **RMSE:** Root Mean Square Error.

Refer to the `results/` directory for detailed logs and generated plots comparing the hybrid model against the standard baseline.

---

## 📂 Project Structure

```plaintext
SAMBA-Finbert/
├── assets/             # Images and architecture diagrams
├── data/               # Data storage (ensure you download datasets here)
├── models/             # PyTorch model definitions
│   ├── samba_block.py  # Core Mamba architecture
│   └── fusion.py       # FinBERT integration logic
├── scripts/            # Utility scripts for preprocessing
├── train.py            # Main training loop (Walk-Forward)
├── evaluate.py         # Evaluation and plotting
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 📄 Citation

If you use this code in your research or find it helpful, please cite the associated paper (currently under review):

```bibtex
@article{sambathkumar2025samba,
  title={SAMBA-FinBERT: Leveraging News Sentiment as a Feature for S\&P 500 Prediction},
  author={Sambathkumar, Sudharshan},
  year={2025},
  journal={arXiv preprint arXiv:24XX.XXXXX}
}
```
