import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import get_crypto_dataset, get_mock_crypto_dataset
from .utils import resolve_technicals_config, set_random_seed
from .components import GCN, LSTM
from .combined_model import AdditiveGraphLSTM, SequentialGraphLSTM


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FIGURE_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, dataset, optimizer, criterion, epochs=2, batch_size=1, dl_kws=None, return_all=False, mode='additive'):
    """
    Train a neural model on chronological crypto samples.

    Args:
        model: nn.Module to train.
        dataset: torch Dataset object containing training samples.
        optimizer: torch optimizer.
        criterion: loss function, usually nn.MSELoss.
        epochs: number of epochs.
        batch_size: DataLoader batch size.
        dl_kws: optional DataLoader keyword arguments.
        return_all: if True, return optimizer too for debugging.
        mode: one of lstm, gcn, additive, or sequential.
    """
    if dl_kws is None:
        dl_kws = {}

    dataloader = DataLoader(dataset, batch_size=batch_size, **dl_kws)
    model.to(device)

    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        print(f'Starting epoch {epoch}...')

        epoch_avg_loss = 0.0
        n_iter = 0
        pbar = tqdm(dataloader)

        for features, target, adj in pbar:
            features = features.float().to(device)
            target = target.float().to(device)
            adj = adj.float().to(device)

            if mode == 'gcn':
                # GCN expects node-level features: (batch_size, n_assets, features_per_asset).
                # Dataset features are flattened as (batch_size, seq_len, n_assets * features_per_asset).
                features = features[:, -1, :].view(features.shape[0], 14, -1)
            else:
                model.initialize_hidden_state(features.shape[0], device=features.device)

            if mode == 'lstm':
                output, hidden_state = model(features)
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            else:
                output = model(features, adj.squeeze())

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_avg_loss += loss.item()
            n_iter += 1
            pbar.set_postfix({'avg loss': epoch_avg_loss / n_iter})

        epoch_losses.append(epoch_avg_loss / n_iter)
        print(f'Epoch {epoch} completed. Avg epoch loss: {epoch_losses[-1]:.4f}')

    if return_all:
        return model, epoch_losses, optimizer
    return model, epoch_losses


def plot_loss(losses, model_name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.savefig(FIGURE_DIR / f"{model_name}_training_loss.pdf", bbox_inches="tight")
    plt.close(fig)


def _create_model(mode, technicals):
    if mode == 'lstm':
        return LSTM(input_size=98 + 14 * len(technicals), hidden_size=14, batch_first=True, predict=True)
    if mode == 'gcn':
        return GCN(n_features=7 + len(technicals), n_pred_per_node=3, predict=True)
    if mode == 'additive':
        return AdditiveGraphLSTM(n_features=7 + len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3)
    if mode == 'sequential':
        return SequentialGraphLSTM(n_features=7 + len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3)
    raise ValueError(f"Unsupported training mode: {mode}")


def _get_dataset(mode, technicals, use_mock_data, seed):
    seq_len = 10

    if use_mock_data:
        dataset = get_mock_crypto_dataset(seq_len=seq_len, technicals=technicals, n_samples=100, seed=seed)
    else:
        dataset = get_crypto_dataset(seq_len=seq_len, technicals=technicals)

    return dataset, seq_len


def _write_training_outputs(model_name, losses, run_config):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / f"{model_name}_training_loss.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    with open(RESULTS_DIR / f"{model_name}_train_config.json", "w") as f:
        json.dump(run_config, f, indent=4)


def main(
    mode,
    technicals,
    technical_names,
    epochs,
    model_name,
    use_mock_data=False,
    seed=42,
    batch_size=1,
    learning_rate=1e-4,
    momentum=0.9,
):
    set_random_seed(seed)

    print('Creating model...')
    model = _create_model(mode, technicals).float()
    print('Model created.\n')

    print('Creating dataset...')
    dataset, seq_len = _get_dataset(mode, technicals, use_mock_data, seed)
    print('Dataset created.\n')

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    run_config = {
        "model_name": model_name,
        "model_type": mode,
        "technicals": technical_names,
        "seq_len": seq_len,
        "epochs": epochs,
        "optimizer": "SGD",
        "learning_rate": learning_rate,
        "momentum": momentum,
        "batch_size": batch_size,
        "seed": seed,
        "use_mock_data": use_mock_data,
        "device": str(device),
    }

    print('Starting training...')
    model, losses = train(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        batch_size=batch_size,
        mode=mode,
    )

    print('Model trained. Saving model...')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(CHECKPOINT_DIR / f"{model_name}.pth"))
    print('Model saved.')

    _write_training_outputs(model_name, losses, run_config)
    plot_loss(losses, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', dest='mode', required=True,
                        choices=['lstm', 'gcn', 'additive', 'sequential'],
                        help='Which model is going to be trained')
    parser.add_argument('--technicals_config', dest='technicals_config', required=True,
                        help='JSON file with mapping of feature names to registered technical indicator names')
    parser.add_argument('--epochs', dest='epochs', required=True, type=int,
                        help='Number of epochs to train model for')
    parser.add_argument('--model_name', dest='model_name', required=True,
                        help='Name for saving model to local directory')
    parser.add_argument('--use_mock_data', action='store_true',
                        help='Use small synthetic dataset instead of Kaggle data')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-4,
                        help='Optimizer learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='SGD momentum')
    args = parser.parse_args()

    with open(args.technicals_config, 'r') as file:
        raw_config = json.load(file)

    technicals = resolve_technicals_config(raw_config)

    main(
        mode=args.mode,
        technicals=technicals,
        technical_names=list(raw_config.keys()),
        epochs=args.epochs,
        model_name=args.model_name,
        use_mock_data=args.use_mock_data,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )
