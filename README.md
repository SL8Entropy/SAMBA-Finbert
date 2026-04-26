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
* **Price Data:** OHLCV data for target indices (e.g. S&P 500).

Run the preprocessing script to tokenize news and generate daily sentiment scores:

```bash
python scripts/preprocess_finbert.py --input data/raw_news.csv --output data/sentiment_scores.csv
```

### 2. Training and Testing

To train and evaluate the models on your dataset:

```bash
python main.py --model samba --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 5
```

**Arguments:**

* `--model`: Choose which model to train: `samba` (proposed model) or `lstm` (baseline). Default is `samba`.
* `--dataset`: The file path to your target dataset CSV (e.g., `Dataset/sp500_with_indicators.csv`). 
* `--num_features`: The number of input parameters/features in your dataset. If not provided, it will attempt to default to the dataset's configuration.
* `--seed`: Random seed for reproducibility. Overrides the default seed in the configuration file.
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
SAMBA-FINBERT/
|   abc.PNG
|   main.py
|   paper_config.py
|   README.md
|   requirements.txt
|   runall.bat
|   test_system.py
|
+---config
|   |   model_config.py
|   |   __init__.py
|   |
|   \---__pycache__
|           model_config.cpython-312.pyc
|           __init__.cpython-312.pyc
|
+---Dataset
|       analyze_sentiment.py
|       calculateIndicators.py
|       create_price_graph.py
|       daily_sentiment_clean.csv
|       merge_sentiment_data.py
|       sp500_headlines_2008_2024.csv
|       sp500_index.csv
|       sp500_index_plot.png
|       sp500_with_indicators.csv
|       sp500_with_indicators_llm.csv
|
+---models
|   |   graph_layers.py
|   |   lstm.py
|   |   mamba.py
|   |   mamba_block.py
|   |   normalization.py
|   |   samba.py
|   |   __init__.py
|   |
|   \---__pycache__
|           graph_layers.cpython-312.pyc
|           lstm.cpython-312.pyc
|           mamba.cpython-312.pyc
|           mamba_block.cpython-312.pyc
|           normalization.cpython-312.pyc
|           samba.cpython-312.pyc
|           __init__.cpython-312.pyc
|
+---results
|       create_graphs.py
|       final_summary_metrics.csv
|       look.py
|       results_lstm_sp500_with_indicators_1.json
|       results_lstm_sp500_with_indicators_1.png
|       results_lstm_sp500_with_indicators_1.txt
|       results_lstm_sp500_with_indicators_1_shap_summary.png
|       results_lstm_sp500_with_indicators_2.json
|       results_lstm_sp500_with_indicators_2.png
|       results_lstm_sp500_with_indicators_2.txt
|       results_lstm_sp500_with_indicators_2_shap_summary.png
|       results_lstm_sp500_with_indicators_3.json
|       results_lstm_sp500_with_indicators_3.png
|       results_lstm_sp500_with_indicators_3.txt
|       results_lstm_sp500_with_indicators_3_shap_summary.png
|       results_lstm_sp500_with_indicators_4.json
|       results_lstm_sp500_with_indicators_4.png
|       results_lstm_sp500_with_indicators_4.txt
|       results_lstm_sp500_with_indicators_4_shap_summary.png
|       results_lstm_sp500_with_indicators_5.json
|       results_lstm_sp500_with_indicators_5.png
|       results_lstm_sp500_with_indicators_5.txt
|       results_lstm_sp500_with_indicators_5_shap_summary.png
|       results_lstm_sp500_with_indicators_llm_1.json
|       results_lstm_sp500_with_indicators_llm_1.png
|       results_lstm_sp500_with_indicators_llm_1.txt
|       results_lstm_sp500_with_indicators_llm_1_shap_summary.png
|       results_lstm_sp500_with_indicators_llm_2.json
|       results_lstm_sp500_with_indicators_llm_2.png
|       results_lstm_sp500_with_indicators_llm_2.txt
|       results_lstm_sp500_with_indicators_llm_2_shap_summary.png
|       results_lstm_sp500_with_indicators_llm_3.json
|       results_lstm_sp500_with_indicators_llm_3.png
|       results_lstm_sp500_with_indicators_llm_3.txt
|       results_lstm_sp500_with_indicators_llm_3_shap_summary.png
|       results_lstm_sp500_with_indicators_llm_4.json
|       results_lstm_sp500_with_indicators_llm_4.png
|       results_lstm_sp500_with_indicators_llm_4.txt
|       results_lstm_sp500_with_indicators_llm_4_shap_summary.png
|       results_lstm_sp500_with_indicators_llm_5.json
|       results_lstm_sp500_with_indicators_llm_5.png
|       results_lstm_sp500_with_indicators_llm_5.txt
|       results_lstm_sp500_with_indicators_llm_5_shap_summary.png
|       results_samba_sp500_with_indicators_1.json
|       results_samba_sp500_with_indicators_1.png
|       results_samba_sp500_with_indicators_1.txt
|       results_samba_sp500_with_indicators_1_shap_summary.png
|       results_samba_sp500_with_indicators_2.json
|       results_samba_sp500_with_indicators_2.png
|       results_samba_sp500_with_indicators_2.txt
|       results_samba_sp500_with_indicators_2_shap_summary.png
|       results_samba_sp500_with_indicators_3.json
|       results_samba_sp500_with_indicators_3.png
|       results_samba_sp500_with_indicators_3.txt
|       results_samba_sp500_with_indicators_3_shap_summary.png
|       results_samba_sp500_with_indicators_4.json
|       results_samba_sp500_with_indicators_4.png
|       results_samba_sp500_with_indicators_4.txt
|       results_samba_sp500_with_indicators_4_shap_summary.png
|       results_samba_sp500_with_indicators_5.json
|       results_samba_sp500_with_indicators_5.png
|       results_samba_sp500_with_indicators_5.txt
|       results_samba_sp500_with_indicators_5_shap_summary.png
|       results_samba_sp500_with_indicators_llm_1.json
|       results_samba_sp500_with_indicators_llm_1.png
|       results_samba_sp500_with_indicators_llm_1.txt
|       results_samba_sp500_with_indicators_llm_1_shap_summary.png
|       results_samba_sp500_with_indicators_llm_2.json
|       results_samba_sp500_with_indicators_llm_2.png
|       results_samba_sp500_with_indicators_llm_2.txt
|       results_samba_sp500_with_indicators_llm_2_shap_summary.png
|       results_samba_sp500_with_indicators_llm_3.json
|       results_samba_sp500_with_indicators_llm_3.png
|       results_samba_sp500_with_indicators_llm_3.txt
|       results_samba_sp500_with_indicators_llm_3_shap_summary.png
|       results_samba_sp500_with_indicators_llm_4.json
|       results_samba_sp500_with_indicators_llm_4.png
|       results_samba_sp500_with_indicators_llm_4.txt
|       results_samba_sp500_with_indicators_llm_4_shap_summary.png
|       results_samba_sp500_with_indicators_llm_5.json
|       results_samba_sp500_with_indicators_llm_5.png
|       results_samba_sp500_with_indicators_llm_5.txt
|       results_samba_sp500_with_indicators_llm_5_shap_summary.png
|
+---trainer
|   |   trainer.py
|   |   __init__.py
|   |
|   \---__pycache__
|           trainer.cpython-312.pyc
|           __init__.cpython-312.pyc
|
+---utils
|   |   data_utils.py
|   |   logger.py
|   |   metrics.py
|   |   model_utils.py
|   |   __init__.py
|   |
|   \---__pycache__
|           data_utils.cpython-312.pyc
|           logger.cpython-312.pyc
|           metrics.cpython-312.pyc
|           model_utils.cpython-312.pyc
|           __init__.cpython-312.pyc
|
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
