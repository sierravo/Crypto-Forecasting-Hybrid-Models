import pytest
import os
import torch
from torch.utils.data import IterableDataset

from crypto_forecasting.data import get_mock_crypto_dataset
from crypto_forecasting.components import GCN, LSTM
from crypto_forecasting.combined_model import AdditiveGraphLSTM, SequentialGraphLSTM
from crypto_forecasting.train import train
from crypto_forecasting.eval import main as eval_main

@pytest.fixture()
def dataset():
    return get_mock_crypto_dataset(seq_len = 10, n_samples = 5)

def test_dataset_loading(dataset):
    assert isinstance(dataset, IterableDataset)

    x, y, adj = next(iter(dataset))

    assert x.shape == (10, 98)      # 14 assets * 7 features each
    assert y.shape == (14,)         # one target per asset
    assert adj.shape == (14, 14)    # asset-by-asset adjacency


def test_lstm_forward():
    model = LSTM(input_size=98, hidden_size=14, batch_first=True, predict=True).float()
    model.initialize_hidden_state(batch_size=1)

    x = torch.randn(1, 10, 98)
    output, hidden = model(x)

    assert output.shape == (1, 14)
    assert hidden[0].shape == (1, 1, 14)
    assert hidden[1].shape == (1, 1, 14)


def test_gcn_forward():
    gcn = GCN(7, 2).float()

    x = torch.randn(14, 7)
    adj = torch.eye(14, dtype=torch.float32)

    output = gcn(x, adj)
    assert output.shape == (1, 14)


def test_additive_forward():
    model = AdditiveGraphLSTM(
        n_features=7,
        lstm_hidden_dim=14,
        gcn_pred_per_node=1
    ).float()

    model.initialize_hidden_state(batch_size=1)

    x = torch.randn(1, 10, 98)
    adj = torch.eye(14, dtype=torch.float32)

    output = model(x, adj)
    assert output.shape == (1, 14)


def test_sequential_forward():
    model = SequentialGraphLSTM(
        n_features=7,
        lstm_hidden_dim=14,
        gcn_pred_per_node=1
    ).float()

    model.initialize_hidden_state(batch_size=1)

    x = torch.randn(1, 10, 98)
    adj = torch.eye(14, dtype=torch.float32)

    output = model(x, adj)
    assert output.shape == (1, 14)


@pytest.mark.parametrize("mode", ["lstm", "gcn", "additive", "sequential"])
def test_eval_smoke_by_mode(mode):
    technicals = {}
    model_name = f"smoke_{mode}"

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if mode == "lstm":
        model = LSTM(input_size=98, hidden_size=14, batch_first=True, predict=True).float()
    elif mode == "gcn":
        model = GCN(n_features=7, n_pred_per_node=1, predict=True).float()
    elif mode == "additive":
        model = AdditiveGraphLSTM(
            n_features=7,
            lstm_hidden_dim=14,
            gcn_pred_per_node=3
        ).float()
    else:
        model = SequentialGraphLSTM(
            n_features=7,
            lstm_hidden_dim=14,
            gcn_pred_per_node=3
        ).float()

    model.save(f"checkpoints/{model_name}.pth")

    eval_main(
        eval_model=mode,
        technicals=technicals,
        model_name=model_name,
        use_mock_data=True
    )

    result_path = f"results/{model_name}_loss.txt"
    assert os.path.exists(result_path)

    with open(result_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > 0

    losses = [float(line) for line in lines]
    assert all(loss >= 0 for loss in losses)

    os.remove(f"checkpoints/{model_name}.pth")
    os.remove(result_path)

