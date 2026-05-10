import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import get_crypto_dataset, get_mock_crypto_dataset
from .utils import compute_regression_metrics, resolve_technicals_config, set_random_seed
from .components import GCN, LSTM
from .combined_model import AdditiveGraphLSTM, SequentialGraphLSTM
from .baselines import BASELINE_NAMES, create_baseline


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_adjacency_mode(adj, adjacency_mode='correlation'):
    """
    Replace or keep an adjacency matrix for graph-structure ablation.

    Args:
        adj: torch.Tensor with shape (n_nodes, n_nodes) or
            (batch_size, n_nodes, n_nodes).
        adjacency_mode: one of correlation, identity, or random.
    """
    if adjacency_mode == 'correlation':
        return adj

    if adj.dim() not in (2, 3):
        raise ValueError(f"adj must have shape (n_nodes, n_nodes) or (batch, n_nodes, n_nodes), got {adj.shape}")

    n_nodes = adj.shape[-1]
    if adj.shape[-2] != n_nodes:
        raise ValueError(f"adj must be square on the last two dimensions, got {adj.shape}")

    if adjacency_mode == 'identity':
        identity = torch.eye(n_nodes, dtype=adj.dtype, device=adj.device)
        if adj.dim() == 3:
            identity = identity.unsqueeze(0).expand(adj.shape[0], -1, -1)
        return identity

    if adjacency_mode == 'random':
        random_adj = torch.rand_like(adj)
        random_adj = (random_adj + random_adj.transpose(-1, -2)) / 2

        identity = torch.eye(n_nodes, dtype=adj.dtype, device=adj.device)
        if adj.dim() == 3:
            identity = identity.unsqueeze(0).expand(adj.shape[0], -1, -1)

        random_adj = random_adj * (1 - identity) + identity
        return random_adj

    raise ValueError("adjacency_mode must be one of: correlation, identity, random")


def build_predictions_dataframe(y_true, y_pred):
    """
    Build a row-level prediction table for post-hoc error analysis.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    if y_true.ndim != 2:
        raise ValueError(f"Expected predictions with 1 or 2 dimensions, got shape {y_true.shape}")

    n_samples, n_assets = y_true.shape
    rows = []

    for sample_index in range(n_samples):
        for asset_index in range(n_assets):
            actual = float(y_true[sample_index, asset_index])
            predicted = float(y_pred[sample_index, asset_index])
            error = predicted - actual
            actual_direction = int(actual > 0)
            predicted_direction = int(predicted > 0)

            rows.append({
                "sample_index": sample_index,
                "asset_index": asset_index,
                "y_true": actual,
                "y_pred": predicted,
                "error": error,
                "abs_error": abs(error),
                "squared_error": error ** 2,
                "actual_direction": actual_direction,
                "predicted_direction": predicted_direction,
                "direction_correct": int(actual_direction == predicted_direction),
            })

    return pd.DataFrame(rows)


def evaluate(model, dataset, criterion, batch_size=1, dl_kws=None, mode='additive', adjacency_mode='correlation'):
    """
    Evaluate a trained neural model and return losses, metrics, and predictions.
    """
    if dl_kws is None:
        dl_kws = {}

    dataloader = DataLoader(dataset, batch_size=batch_size, **dl_kws)
    model.to(device)
    model.eval()

    losses = []
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for features, target, adj in tqdm(dataloader):
            features = features.float().to(device)
            target = target.float().to(device)
            adj = adj.float().to(device)

            if mode in {'gcn', 'additive', 'sequential'}:
                adj = apply_adjacency_mode(adj, adjacency_mode)

            if mode != 'gcn':
                model.initialize_hidden_state(features.shape[0], device=features.device)

            if mode == 'lstm':
                output, hidden_state = model(features)
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            elif mode == 'gcn':
                features = features[:, -1, :].view(features.shape[0], 14, -1)
                output = model(features, adj.squeeze())
            else:
                output = model(features, adj.squeeze())

            loss = criterion(output, target)
            losses.append(loss.item())
            all_outputs.append(output.detach().cpu())
            all_targets.append(target.detach().cpu())

    y_pred = torch.cat(all_outputs, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    metrics = compute_regression_metrics(y_true, y_pred)
    predictions_df = build_predictions_dataframe(y_true, y_pred)

    return losses, metrics, predictions_df


def evaluate_baseline(baseline, dataset, criterion, dl_kws=None):
    """
    Evaluate a non-trainable baseline on chronological samples.

    The baseline predicts first, then updates from the current target. This keeps
    previous-target and rolling-mean baselines from leaking the current answer.
    """
    if dl_kws is None:
        dl_kws = {}

    dataloader = DataLoader(dataset, batch_size=1, **dl_kws)
    baseline.reset()

    losses = []
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for features, target, adj in tqdm(dataloader):
            features = features.float()
            target = target.float()
            adj = adj.float()

            output = baseline.predict(features, target, adj)
            loss = criterion(output, target)
            losses.append(loss.item())

            all_outputs.append(output.detach().cpu())
            all_targets.append(target.detach().cpu())
            baseline.update(target)

    y_pred = torch.cat(all_outputs, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    metrics = compute_regression_metrics(y_true, y_pred)
    predictions_df = build_predictions_dataframe(y_true, y_pred)

    return losses, metrics, predictions_df


def _write_evaluation_outputs(model_name, losses, metrics, predictions_df, run_config):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / f"{model_name}_loss.txt", "w") as f:
        for loss in losses:
            f.write(f'{loss}\n')

    with open(RESULTS_DIR / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    predictions_df.to_csv(RESULTS_DIR / f"{model_name}_predictions.csv", index=False)

    with open(RESULTS_DIR / f"{model_name}_eval_config.json", "w") as f:
        json.dump(run_config, f, indent=4)


def _print_metrics(model_name, losses, metrics):
    print('Average batch/sample MSE for {}: {:.8f}'.format(model_name, np.mean(losses)))
    print('Evaluation metrics for {}:'.format(model_name))
    for metric_name in ['mse', 'rmse', 'mae', 'directional_accuracy']:
        print('  {}: {:.8f}'.format(metric_name, metrics[metric_name]))


def _get_dataset(eval_model, technicals, use_mock_data, seed):
    seq_len = 10

    if use_mock_data:
        dataset = get_mock_crypto_dataset(seq_len=seq_len, technicals=technicals, evaluation=True, n_samples=30, seed=seed)
    else:
        dataset = get_crypto_dataset(seq_len=seq_len, technicals=technicals, evaluation=True)

    return dataset, seq_len


def _create_model(eval_model, technicals):
    if eval_model == 'lstm':
        return LSTM(input_size=98 + 14 * len(technicals), hidden_size=14, batch_first=True, predict=True)
    if eval_model == 'gcn':
        return GCN(n_features=7 + len(technicals), n_pred_per_node=3, predict=True)
    if eval_model == 'additive':
        return AdditiveGraphLSTM(n_features=7 + len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3)
    if eval_model == 'sequential':
        return SequentialGraphLSTM(n_features=7 + len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3)
    raise ValueError(f"Unsupported model type: {eval_model}")


def main(
    eval_model,
    technicals,
    technical_names=None,
    model_name=None,
    use_mock_data=False,
    rolling_window=5,
    adjacency_mode='correlation',
    seed=42,
    batch_size=1,
    checkpoint_name=None,
):
    if model_name is None:
        raise ValueError("model_name is required")
    if technical_names is None:
        technical_names = list(technicals.keys())
    if checkpoint_name is None:
        checkpoint_name = model_name

    set_random_seed(seed)

    print('Creating dataset...')
    dataset, seq_len = _get_dataset(eval_model, technicals, use_mock_data, seed)
    print('Dataset created.\n')

    criterion = nn.MSELoss()

    run_config = {
        "model_name": model_name,
        "eval_model": eval_model,
        "technicals": technical_names,
        "seq_len": seq_len,
        "rolling_window": rolling_window if eval_model == 'rolling_mean' else None,
        "adjacency_mode": adjacency_mode if eval_model in {'gcn', 'additive', 'sequential'} else None,
        "batch_size": batch_size if eval_model not in BASELINE_NAMES else 1,
        "seed": seed,
        "use_mock_data": use_mock_data,
        "device": str(device),
        "checkpoint_name": checkpoint_name if eval_model not in BASELINE_NAMES else None,
    }

    if eval_model in BASELINE_NAMES:
        print('Creating baseline...')
        baseline = create_baseline(eval_model, rolling_window=rolling_window)
        print('Baseline created.\n')
        losses, metrics, predictions_df = evaluate_baseline(baseline, dataset, criterion)
    else:
        print('Creating model...')
        model = _create_model(eval_model, technicals)
        checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pth"
        model.load(str(checkpoint_path), map_location=device)
        model.float()
        print('Model created.\n')
        losses, metrics, predictions_df = evaluate(
            model,
            dataset,
            criterion,
            batch_size=batch_size,
            mode=eval_model,
            adjacency_mode=adjacency_mode,
        )

    _write_evaluation_outputs(model_name, losses, metrics, predictions_df, run_config)
    _print_metrics(model_name, losses, metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_model', dest='eval_model', required=True,
                        choices=['lstm', 'gcn', 'additive', 'sequential', 'zero', 'previous_target', 'rolling_mean'],
                        help='Which model or baseline is going to be evaluated')
    parser.add_argument('--technicals_config', dest='technicals_config', required=True,
                        help='JSON file with mapping of feature names to registered technical indicator names')
    parser.add_argument('--model_name', dest='model_name', required=True,
                        help='Name for loading model from / writing results to local directory')
    parser.add_argument('--use_mock_data', action='store_true',
                        help='Use small synthetic dataset instead of Kaggle data')
    parser.add_argument('--rolling_window', dest='rolling_window', type=int, default=5,
                        help='Number of previous target vectors used by the rolling_mean baseline')
    parser.add_argument('--adjacency_mode', dest='adjacency_mode', default='correlation',
                        choices=['correlation', 'identity', 'random'],
                        help='Graph ablation: correlation uses dataset adjacency, identity removes cross-asset edges, random uses a symmetric random graph')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1,
                        help='Evaluation batch size for neural models. Baselines are evaluated chronologically with batch_size=1.')
    parser.add_argument('--checkpoint_name', dest='checkpoint_name', default=None,
                        help='Checkpoint filename stem to load. Defaults to model_name. Useful when evaluating the same checkpoint under multiple adjacency ablations.')
    args = parser.parse_args()

    with open(args.technicals_config, 'r') as file:
        raw_config = json.load(file)

    technicals = resolve_technicals_config(raw_config)

    main(
        eval_model=args.eval_model,
        technicals=technicals,
        technical_names=list(raw_config.keys()),
        model_name=args.model_name,
        use_mock_data=args.use_mock_data,
        rolling_window=args.rolling_window,
        adjacency_mode=args.adjacency_mode,
        seed=args.seed,
        batch_size=args.batch_size,
        checkpoint_name=args.checkpoint_name,
    )
