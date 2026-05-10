# Cryptocurrency Forecasting with Sequence, Graph, and Hybrid Neural Networks

This project investigates short-horizon cryptocurrency forecasting across multiple assets using sequence models, graph models, and hybrid sequence–graph neural networks. I built an end-to-end research pipeline for preprocessing the G-Research Crypto Forecasting dataset, generating technical features, training multiple neural architectures, and evaluating whether added model complexity improves predictive performance over simpler alternatives.

The main outcome is a reproducible model-comparison framework for testing whether temporal modeling, cross-asset graph structure, or hybrid architectures add value in multi-asset crypto forecasting.

## Decision Question

This project is framed as a model-selection problem:

> Does adding graph-based cross-asset structure improve short-horizon cryptocurrency forecasting enough to justify the added complexity over simpler baselines?

The primary deliverable is **not** a trading system. It is a comparative ML research pipeline designed to support reproducible evaluation, baseline comparison, and architecture-level decisions.

## Problem Context

Cryptocurrency markets are noisy, non-stationary, and influenced by both asset-specific time dynamics and broader cross-asset relationships.

- Sequence models such as LSTMs can capture temporal information within historical windows.
- Graph models such as GCNs can represent relationships between assets through an adjacency matrix.
- Hybrid models can combine both approaches, but their additional complexity is only useful if it produces measurable gains over simpler benchmarks.

This project tests whether that extra complexity is warranted.

## What This Repository Contains

A research-style machine learning repository with:

- preprocessing for multi-asset crypto time series
- configurable technical-indicator feature generation
- chronological train/evaluation splitting
- feature normalization fitted only on the training split
- four neural model variants
- simple and diagnostic baselines
- adjacency ablation support
- evaluation metrics including MSE, RMSE, MAE, directional accuracy, and per-asset metrics
- saved prediction-level outputs for later error analysis
- mock-data support for smoke testing without the Kaggle dataset
- a lightweight automated test suite

## Project Origin

Originally developed as a graduation research project on crypto forecasting, then refactored and extended for reproducibility, stronger evaluation, and portfolio presentation.

## My Contributions

- Refactored the repository structure for clarity and reproducibility
- Fixed training and evaluation inconsistencies
- Added mock-data support for end-to-end smoke testing
- Rewrote tests to remove dependency on private Kaggle files
- Added safer preprocessing, feature normalization, and NaN/inf handling
- Added reusable evaluation metrics and per-asset metrics
- Added baseline models for comparison
- Added identity and random adjacency ablations
- Added prediction saving, run-config logging, and results summarization
- Improved documentation, setup, and reproducibility
- Cleaned model implementation details and path handling
- Packaged the project for portfolio presentation

## Dataset Summary

This project uses the **G-Research Crypto Forecasting** dataset from Kaggle, which contains historical market data for multiple cryptocurrency assets.

- **Source:** G-Research Crypto Forecasting competition dataset
- **Domain:** multi-asset financial time series
- **Assets used:** 14
- **Inputs:** historical market features plus optional technical indicators
- **Targets:** one target value per asset for each training example

Because the raw dataset is too large to include in the repository, it is not tracked in Git and must be downloaded separately.

### Expected Local Data Location

```text
data/g-research-crypto-forecasting/
```

At minimum, the code expects:

```text
data/g-research-crypto-forecasting/train.csv
data/g-research-crypto-forecasting/asset_details.csv
```

The preprocessing pipeline may also generate cached intermediate files such as:

```text
data/filtered_features.csv
data/filtered_targets.csv
data/filtered_log_returns.csv
```

## Data Pipeline

The data pipeline:

1. sorts records chronologically by timestamp and asset
2. builds one aligned multi-asset feature matrix
3. generates optional technical indicators from a configuration file
4. computes rolling correlation-based adjacency matrices from historical log returns
5. fills missing values and removes invalid numeric values
6. fits normalization statistics on the training split only
7. applies the same training-derived normalization to both train and evaluation data
8. yields rolling windows of features, asset-level targets, and adjacency matrices

Each sample contains:

- `features`: a rolling window of historical multi-asset inputs
- `target`: one target value per asset
- `adj`: an asset-by-asset adjacency matrix

## Models Compared

### 1. LSTM

A temporal baseline that processes rolling windows of flattened multi-asset features.

### 2. GCN

A graph-based model that uses an asset adjacency matrix to aggregate information across assets.

### 3. Additive Graph + LSTM Hybrid

A hybrid architecture that combines separate LSTM and GCN predictions through a learned weighted average.

### 4. Sequential Graph + LSTM Hybrid

A hybrid architecture that first produces per-asset sequence embeddings and then passes those embeddings through a graph layer before final prediction.

## Evaluation Design

The final evaluation uses:

- a chronological holdout split
- 14 assets
- sequence length of 10
- fixed seed `42`
- shared evaluation metrics across all models
- saved prediction-level outputs for later analysis

### Metrics

The pipeline reports:

- MSE
- RMSE
- MAE
- directional accuracy
- per-asset MSE / RMSE / MAE / directional accuracy

### Adjacency Ablation

To test whether graph structure contributes meaningful signal, the sequential hybrid is evaluated under three adjacency settings:

- **Correlation adjacency:** graph edges based on rolling historical return correlations
- **Identity adjacency:** no cross-asset information; self-connections only
- **Random adjacency:** synthetic symmetric graph used as a negative control

## Main Results

### Deployable Model Comparison

These runs use information available from the model input pipeline at prediction time.

| Model | RMSE | MAE | Directional Accuracy | Complexity | Decision |
|---|---:|---:|---:|---|---|
| **Zero baseline** | **0.00378** | **0.00224** | N/A | Very low | **Best deployable benchmark tested** |
| Sequential Hybrid + random adjacency | 0.00463 | 0.00300 | 50.39% | High | Worse than zero baseline |
| Sequential Hybrid + identity adjacency | 0.00464 | 0.00301 | 50.41% | High | Worse than zero baseline |
| Sequential Hybrid + correlation adjacency | 0.00465 | 0.00302 | 50.39% | High | Correlation graph adds no measurable value |
| Additive Hybrid + correlation adjacency | 0.01086 | 0.00489 | 50.89% | High | Underperforms |
| GCN + correlation adjacency | 0.01139 | 0.00505 | 50.92% | Medium | Underperforms |
| LSTM | 0.01534 | 0.00613 | 50.43% | Medium | Underperforms |

### Diagnostic Target-History Baselines

These baselines are useful for understanding the target series, but they are not treated as deployable live benchmarks because they rely on prior target labels rather than only observable model inputs.

| Diagnostic Baseline | RMSE | MAE | Directional Accuracy | Interpretation |
|---|---:|---:|---:|---|
| Previous target | **0.00195** | **0.00095** | **86.47%** | Indicates strong short-lag target persistence |
| Rolling mean of prior targets | 0.00231 | 0.00132 | 80.31% | Also indicates target persistence |

## Findings

### 1. Added model complexity was not justified in the current setup

The zero baseline outperformed every deployable neural model tested. Under the current feature set and evaluation design, the LSTM, GCN, and hybrid models did not produce enough incremental value to justify their added complexity.

### 2. Correlation-based graph structure did not improve performance

The sequential hybrid produced nearly identical results under correlation, identity, and random adjacency settings:

| Adjacency Type | RMSE |
|---|---:|
| Random | 0.00463 |
| Identity | 0.00464 |
| Correlation | 0.00465 |

Because performance is essentially unchanged across adjacency types, the correlation graph did not add measurable predictive value in this experiment.

### 3. Diagnostic target-history baselines revealed strong persistence

The previous-target and rolling-target baselines performed much better than the neural models, suggesting strong short-lag persistence in the target series. However, because these baselines use prior target labels rather than only currently observable model inputs, they are retained as diagnostic references rather than as deployable benchmarks.

### 4. Baseline design changed the conclusion

The earlier version of this project focused only on neural-model comparison. Adding simple baselines and adjacency ablations changed the conclusion from “hybrid models perform best” to a more decision-relevant result:

> In the current setup, simpler alternatives outperform the tested neural architectures, and graph structure does not provide measurable benefit.

## Business Interpretation

Although this project is not a trading system, the pipeline is relevant to several financial-analytics use cases:

- **Risk monitoring:** identify whether assets move together during periods of market stress and whether cross-asset structure improves short-horizon forecasting.
- **Asset clustering:** use correlation structure and learned representations to study groups of assets with similar behavior.
- **Signal research:** test whether temporal features, technical indicators, or graph relationships contain incremental predictive value before investing in more complex modeling.
- **Anomaly detection:** compare observed asset behavior with learned temporal or cross-asset patterns to surface unusual movements.
- **Feature discovery:** evaluate whether engineered indicators or graph-derived relationships contribute useful signal beyond simple benchmarks.

The practical value of the project is the evaluation framework itself: it helps decide whether a more complex modeling approach is worth further development before committing engineering or research resources.

## Recommendation

Based on the current evidence:

- Do **not** prefer the graph or hybrid neural architectures for deployment under this setup.
- Use the zero baseline as the best deployable benchmark tested so far.
- Treat the target-history baselines as diagnostics, not deployment candidates.
- Before further neural-model development, add stronger valid low-complexity benchmarks using only observable inputs, such as:
  - previous realized return
  - rolling mean of realized returns
  - linear or ridge regression baselines

A more complex model should only be preferred if it consistently beats these valid low-maintenance alternatives.

## Limitations

- The project evaluates forecasting models, but it does **not** test a complete trading strategy, portfolio construction method, or transaction-cost-aware decision rule.
- The final evaluation uses a single chronological holdout split rather than full walk-forward validation across multiple market regimes.
- The neural models were compared under one fixed training setup; the results do not establish that the architectures are globally optimal or fully tuned.
- Correlation-based adjacency is only one way to represent asset relationships and may be unstable in noisy, rapidly changing markets.
- The current deployable baseline set is still incomplete. Stronger valid low-complexity benchmarks based only on observable inputs, such as realized-return baselines or ridge regression, should be added before further model-selection claims.
- Diagnostic target-history baselines reveal target persistence, but they are not valid live deployment benchmarks because prior target labels are not available at the prediction timestamp in this forecasting setup.
- The results are specific to the selected dataset slice, feature set, preprocessing choices, and evaluation design; they should not be generalized to all crypto markets or all forecasting horizons without further testing.

## Key Takeaways

- Strong evaluation requires simple baselines, not only complex model comparisons.
- Graph structure should be tested with ablations rather than assumed to help.
- A model that is more sophisticated is not automatically more useful.
- Chronological validation, leakage-aware preprocessing, and baseline design materially affect conclusions in time-series work.
- The most important outcome of this project is not that a hybrid model won, but that the evaluation framework was strong enough to show when added complexity was not justified.

## Reproducible Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Kaggle dataset

Download the G-Research Crypto Forecasting dataset and place it here:

```text
data/g-research-crypto-forecasting/
```

### 4. Train a model

Example:

```bash
python -m crypto_forecasting.train \
  --training_mode lstm \
  --technicals_config technicals_config.json \
  --epochs 5 \
  --model_name lstm_run \
  --seed 42
```

### 5. Evaluate a trained model

Example:

```bash
python -m crypto_forecasting.eval \
  --eval_model lstm \
  --technicals_config technicals_config.json \
  --model_name lstm_run \
  --seed 42
```

### 6. Evaluate adjacency ablations

```bash
python -m crypto_forecasting.eval \
  --eval_model sequential \
  --technicals_config technicals_config.json \
  --model_name sequential_identity \
  --adjacency_mode identity \
  --seed 42
```

```bash
python -m crypto_forecasting.eval \
  --eval_model sequential \
  --technicals_config technicals_config.json \
  --model_name sequential_random \
  --adjacency_mode random \
  --seed 42
```

### 7. Summarize results

```bash
python -m crypto_forecasting.summarize_results
```

This creates:

```text
results/summary_metrics.csv
```

### 8. Run tests

```bash
python -m pytest
```

## Mock Dataset for Smoke Testing

This repository includes a small synthetic dataset for smoke testing, so the training and evaluation pipeline can be run without downloading the full Kaggle dataset.

The mock dataset is designed to match the interface of the real dataset and yields:

- `features`: sequence input tensor
- `target`: one target value per asset
- `adj`: adjacency matrix across assets

This is intended only for quick validation that the code runs end to end. It is **not** meant for meaningful model performance evaluation.

### Run a Quick Smoke Test

Train with mock data:

```bash
python -m crypto_forecasting.train \
  --training_mode lstm \
  --technicals_config technicals_config.json \
  --epochs 1 \
  --model_name smoke_lstm \
  --use_mock_data \
  --seed 42
```

Evaluate with mock data:

```bash
python -m crypto_forecasting.eval \
  --eval_model lstm \
  --technicals_config technicals_config.json \
  --model_name smoke_lstm \
  --use_mock_data \
  --seed 42
```

You can replace `lstm` with `gcn`, `additive`, or `sequential` to test other model paths.

## Project Structure

```text
crypto-forecasting-gnn/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE.md
│
├── src/
│   └── crypto_forecasting/
│       ├── __init__.py
│       ├── train.py
│       ├── eval.py
│       ├── data.py
│       ├── utils.py
│       ├── baselines.py
│       ├── summarize_results.py
│       ├── components.py
│       ├── combined_model.py
│       └── technicals_config.json
│
├── tests/
│   └── tests.py
│
├── scripts/
│   └── run_evaluations.sh
│
├── checkpoints/              # optional local output, usually gitignored
├── figures/                  # optional local output, usually gitignored
├── results/                  # optional local output, usually gitignored
├── tex/                      # paper resulting from project before updates
├── references/               # reference papers
│
└── data/                     # local only, gitignored
    └── g-research-crypto-forecasting/
```

## License

MIT License. See the `LICENSE` file for details.
