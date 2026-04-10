# Cryptocurrency Forecasting with Sequence, Graph, and Hybrid Neural Networks

This project investigates short-horizon cryptocurrency forecasting using deep learning models that capture both temporal patterns within individual assets and relationships across assets. I built an end-to-end research pipeline for preprocessing the G-Research Crypto Forecasting dataset, generating technical features, training multiple neural architectures, and evaluating their predictive performance. The main outcome was a comparative framework for testing whether hybrid sequence–graph models can outperform single-model baselines on multi-asset crypto forecasting.

## Problem Statement

Cryptocurrency prices are noisy, non-stationary, and influenced by both time-dependent behavior and cross-asset interactions. A standard sequence model such as an LSTM can capture temporal information, but it does not explicitly model relationships between assets. A graph-based model can represent asset-to-asset structure, but may not capture sequential dynamics as effectively on its own.

This project asks: can hybrid models that combine sequence modeling and graph structure better forecast asset-level targets than standalone sequence or graph models?

## What This Repo Contains

 A research-style machine learning repository with:

- a preprocessing pipeline for multi-asset crypto time series
- configurable feature generation using a technical indicators config file
- four model variants for comparison
- training and evaluation scripts
- mock-data support for smoke testing without the Kaggle dataset
- a lightweight test suite for dataset and model sanity checks

The primary deliverable is not a trading system. It is a comparative modeling pipeline for experimentation and reproducible evaluation.

## Project origin

Originally developed as a graduation research project on crypto forecasting.

## My contributions

- Refactored repository structure for clarity and reproducibility
- Fixed training/evaluation inconsistencies
- Added mock-data support for smoke testing
- Rewrote tests to remove dependency on private Kaggle files
- Improved documentation, setup, and reproducibility
- Cleaned model implementation details and path handling
- Packaged the project for portfolio presentation

## Dataset Summary

This project uses the **G-Research Crypto Forecasting** dataset from Kaggle, which contains historical market data for multiple crypto assets.

- Source: G-Research Crypto Forecasting competition dataset
- Domain: multi-asset financial time series
- Assets: 14 assets used in this project
- Inputs: historical market features and optional technical indicators
- Targets: one target value per asset for each training example

Because the raw dataset is too large to include in the repository, it is not tracked in Git and must be downloaded separately.

### Expected local data location

```text
data/g-research-crypto-forecasting/
```

At minimum, the code expects:
```text
data/g-research-crypto-forecasting/train.csv
data/g-research-crypto-forecasting/asset_details.csv
```

The preprocessing pipeline may also generate cached intermediate files such as::
```text
data/filtered_features.csv
data/filtered_targets.csv
data/filtered_log_returns.csv
```
## Methods Compared

This repository compares four model types:

1. LSTM
A sequence model used as a temporal baseline for forecasting from rolling windows of historical asset features.

2. GCN
A graph-based model that uses an adjacency matrix to represent relationships between assets and perform adjacency-weighted aggregation.

3. Additive Graph + LSTM Hybrid
A hybrid model that combines outputs from a graph model and an LSTM through a learned weighted combination.

4. Sequential Graph + LSTM Hybrid
A hybrid model that applies sequence modeling and graph modeling in sequence rather than combining predictions additively.

## Results

| Model             |   Metric |  Result | Notes                                   |
| ----------------- | -------- | ------- | --------------------------------------- |
| LSTM              |    MSE   |   2.18  | Temporal baseline                       |
| Additive Hybrid   |    MSE   |   1.64  | Weighted graph + sequence combination   |
| Sequential Hybrid |    MSE   |   0.14  | Sequence and graph modeling in sequence |

## Key Takeaways

- Temporal and cross-asset structure can be modeled separately and compared within a shared pipeline.
- Hybrid architectures are more flexible than single-model baselines, but they are also more sensitive to implementation details and reproducibility issues.
- Data pipeline consistency, input shaping, and model configuration matter as much as architecture choice in time-series research code.
- The most important outcome of this project was building a working comparative framework for sequence, graph, and hybrid forecasting models on a multi-asset dataset.

## Reproducible Setup

1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the Kaggle dataset

Download the G-Research Crypto Forecasting dataset and place it here:
```text
data/g-research-crypto-forecasting/
```
4. Train a model

Example:
```bash
python crypto_forecasting.train --training_mode lstm --technicals_config technicals_config.json --
```

5. Evaluate
```bash
python -m crypto_forecasting.eval --eval_model lstm --technicals_config src/crypto_forecasting/technicals_config.json --model_name smoke_lstm --use_mock_data
```

5. Testing

```bash
python -m pytest
```

### Mock Dataset for Smoke Testing

This repository includes a small synthetic dataset for smoke testing, so the training and evaluation pipeline can be run without downloading the full Kaggle dataset.

The mock dataset is designed to match the interface of the real dataset and yields:

- `features`: sequence input tensor
- `target`: one target value per asset
- `adj`: adjacency matrix across assets

This is intended only for quick validation that the code runs end to end. It is **not** meant for meaningful model performance evaluation.

### Run a quick smoke test

Train with mock data:

```bash
python train.py --training_mode lstm --technicals_config technicals_config.json --epochs 1 --model_name smoke_lstm --use_mock_data
```

Evaluate with mock data:

```bash
python eval.py --eval_model lstm --technicals_config technicals_config.json --model_name smoke_lstm --use_mock_data
```

You can replace lstm with gcn, additive, or sequential to test other model paths.

## Project Structure

crypto-forecasting-gnn/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/
│   └── crypto_forecasting/
│       ├── __init__.py
│       ├── train.py
│       ├── eval.py
│       ├── data.py
│       ├── utils.py
│       ├── components.py
│       ├── combined_model.py
│       └── technicals_config.json
│
├── tests/
│   └── tests.py  
│
├── checkpoints/              # optional local output, usually gitignored
├── figures/                  # optional local output, usually gitignored
├── results/                  # optional local output, usually gitignored
├── tex/  					  # paper resulting from project
├── references/     		  # reference papers
│
└── data/                     # local only, gitignored
    └── g-research-crypto-forecasting/

## License

MIT License. See the `LICENSE` file for details.




