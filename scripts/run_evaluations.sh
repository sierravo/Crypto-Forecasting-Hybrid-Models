#!/usr/bin/env bash
set -euo pipefail

TECHNICALS_CONFIG=${TECHNICALS_CONFIG:-technicals_config.json}
SEED=${SEED:-42}
ROLLING_WINDOW=${ROLLING_WINDOW:-5}

# Baselines do not require checkpoints.
python -m crypto_forecasting.eval --eval_model zero --technicals_config "$TECHNICALS_CONFIG" --model_name baseline_zero --seed "$SEED"
python -m crypto_forecasting.eval --eval_model previous_target --technicals_config "$TECHNICALS_CONFIG" --model_name baseline_previous_target --seed "$SEED"
python -m crypto_forecasting.eval --eval_model rolling_mean --technicals_config "$TECHNICALS_CONFIG" --model_name baseline_rolling_mean --rolling_window "$ROLLING_WINDOW" --seed "$SEED"

# Neural-model evaluations require matching checkpoints.
python -m crypto_forecasting.eval --eval_model lstm --technicals_config "$TECHNICALS_CONFIG" --model_name lstm_run --seed "$SEED"
python -m crypto_forecasting.eval --eval_model gcn --technicals_config "$TECHNICALS_CONFIG" --model_name gcn_corr --checkpoint_name gcn_run --adjacency_mode correlation --seed "$SEED"
python -m crypto_forecasting.eval --eval_model additive --technicals_config "$TECHNICALS_CONFIG" --model_name additive_corr --checkpoint_name additive_run --adjacency_mode correlation --seed "$SEED"

# Graph ablations. These reuse the same sequential checkpoint and only change adjacency during evaluation.
python -m crypto_forecasting.eval --eval_model sequential --technicals_config "$TECHNICALS_CONFIG" --model_name sequential_corr --checkpoint_name sequential_run --adjacency_mode correlation --seed "$SEED"
python -m crypto_forecasting.eval --eval_model sequential --technicals_config "$TECHNICALS_CONFIG" --model_name sequential_identity --checkpoint_name sequential_run --adjacency_mode identity --seed "$SEED"
python -m crypto_forecasting.eval --eval_model sequential --technicals_config "$TECHNICALS_CONFIG" --model_name sequential_random --checkpoint_name sequential_run --adjacency_mode random --seed "$SEED"

python -m crypto_forecasting.summarize_results
