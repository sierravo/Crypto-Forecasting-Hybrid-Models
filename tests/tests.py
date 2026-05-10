import pytest
import os
import json
import torch
from torch.utils.data import IterableDataset

from crypto_forecasting.data import get_mock_crypto_dataset
from crypto_forecasting.components import GCN, LSTM
from crypto_forecasting.combined_model import AdditiveGraphLSTM, SequentialGraphLSTM
from crypto_forecasting.train import train
from crypto_forecasting.eval import CHECKPOINT_DIR, RESULTS_DIR, apply_adjacency_mode, build_predictions_dataframe, main as eval_main
from crypto_forecasting.utils import compute_regression_metrics
from crypto_forecasting.baselines import ZeroBaseline, PreviousTargetBaseline, RollingMeanTargetBaseline, create_baseline

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
    gcn = GCN(n_features=7, n_pred_per_node=2, predict=True).float()

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


def test_apply_adjacency_mode_identity():
    adj = torch.randn(2, 14, 14)

    identity_adj = apply_adjacency_mode(adj, "identity")
    expected = torch.eye(14).unsqueeze(0).expand(2, -1, -1)

    assert torch.equal(identity_adj, expected)


def test_apply_adjacency_mode_random():
    adj = torch.zeros(2, 14, 14)

    random_adj = apply_adjacency_mode(adj, "random")

    assert random_adj.shape == adj.shape
    assert torch.allclose(random_adj, random_adj.transpose(-1, -2))
    assert torch.allclose(
        torch.diagonal(random_adj, dim1=-2, dim2=-1),
        torch.ones(2, 14),
    )


def test_apply_adjacency_mode_correlation_keeps_input():
    adj = torch.randn(14, 14)

    correlation_adj = apply_adjacency_mode(adj, "correlation")

    assert correlation_adj is adj


@pytest.mark.parametrize("mode", ["lstm", "gcn", "additive", "sequential"])
def test_eval_smoke_by_mode(mode):
    technicals = {}
    model_name = f"smoke_{mode}"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "lstm":
        model = LSTM(input_size=98, hidden_size=14, batch_first=True, predict=True).float()
    elif mode == "gcn":
        model = GCN(n_features=7, n_pred_per_node=3, predict=True).float()
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

    model.save(str(CHECKPOINT_DIR / f"{model_name}.pth"))

    eval_main(
        eval_model=mode,
        technicals=technicals,
        model_name=model_name,
        use_mock_data=True
    )

    result_path = RESULTS_DIR / f"{model_name}_loss.txt"
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    predictions_path = RESULTS_DIR / f"{model_name}_predictions.csv"

    assert os.path.exists(result_path)
    assert os.path.exists(metrics_path)
    assert os.path.exists(predictions_path)

    with open(result_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > 0

    losses = [float(line) for line in lines]
    assert all(loss >= 0 for loss in losses)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "directional_accuracy" in metrics
    assert "per_asset_mae" in metrics

    with open(predictions_path, "r") as f:
        header = f.readline().strip().split(",")

    assert "sample_index" in header
    assert "asset_index" in header
    assert "y_true" in header
    assert "y_pred" in header
    assert "abs_error" in header
    assert "direction_correct" in header

    os.remove(CHECKPOINT_DIR / f"{model_name}.pth")
    os.remove(result_path)
    os.remove(metrics_path)
    os.remove(predictions_path)


@pytest.mark.parametrize("adjacency_mode", ["identity", "random"])
def test_eval_smoke_with_adjacency_ablation(adjacency_mode):
    technicals = {}
    model_name = f"smoke_gcn_{adjacency_mode}"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = GCN(n_features=7, n_pred_per_node=3, predict=True).float()
    model.save(str(CHECKPOINT_DIR / f"{model_name}.pth"))

    eval_main(
        eval_model="gcn",
        technicals=technicals,
        model_name=model_name,
        use_mock_data=True,
        adjacency_mode=adjacency_mode,
    )

    result_path = RESULTS_DIR / f"{model_name}_loss.txt"
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    predictions_path = RESULTS_DIR / f"{model_name}_predictions.csv"

    assert os.path.exists(result_path)
    assert os.path.exists(metrics_path)
    assert os.path.exists(predictions_path)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "mse" in metrics
    assert "per_asset_mae" in metrics

    os.remove(CHECKPOINT_DIR / f"{model_name}.pth")
    os.remove(result_path)
    os.remove(metrics_path)
    os.remove(predictions_path)


@pytest.mark.parametrize("baseline_name", ["zero", "previous_target", "rolling_mean"])
def test_eval_smoke_by_baseline(baseline_name):
    technicals = {}
    model_name = f"smoke_{baseline_name}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    eval_main(
        eval_model=baseline_name,
        technicals=technicals,
        model_name=model_name,
        use_mock_data=True,
    )

    result_path = RESULTS_DIR / f"{model_name}_loss.txt"
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    predictions_path = RESULTS_DIR / f"{model_name}_predictions.csv"

    assert os.path.exists(result_path)
    assert os.path.exists(metrics_path)
    assert os.path.exists(predictions_path)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "directional_accuracy" in metrics
    assert "per_asset_mae" in metrics

    os.remove(result_path)
    os.remove(metrics_path)
    os.remove(predictions_path)


def test_baseline_factory_and_predictions():
    target = torch.tensor([[1.0, -2.0, 3.0]])

    zero = create_baseline("zero")
    assert isinstance(zero, ZeroBaseline)
    assert torch.equal(zero.predict(None, target), torch.zeros_like(target))

    previous = create_baseline("previous_target")
    assert isinstance(previous, PreviousTargetBaseline)
    assert torch.equal(previous.predict(None, target), torch.zeros_like(target))
    previous.update(target)
    assert torch.equal(previous.predict(None, target), target)

    rolling = create_baseline("rolling_mean", rolling_window=2)
    assert isinstance(rolling, RollingMeanTargetBaseline)
    assert torch.equal(rolling.predict(None, target), torch.zeros_like(target))
    rolling.update(torch.tensor([[1.0, 1.0, 1.0]]))
    rolling.update(torch.tensor([[3.0, 3.0, 3.0]]))
    assert torch.equal(rolling.predict(None, target), torch.tensor([[2.0, 2.0, 2.0]]))


def test_train_smoke_lstm():
    dataset = get_mock_crypto_dataset(seq_len=10, n_samples=5)

    model = LSTM(input_size=98, hidden_size=14, batch_first=True, predict=True).float()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    criterion = torch.nn.MSELoss()

    trained_model, losses = train(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        criterion=criterion,
        epochs=1,
        mode="lstm",
    )

    assert len(losses) == 1
    assert losses[0] >= 0

def test_train_smoke_gcn():
    dataset = get_mock_crypto_dataset(seq_len=1, n_samples=5)

    model = GCN(n_features=7, n_pred_per_node=1, predict=True).float()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    criterion = torch.nn.MSELoss()

    trained_model, losses = train(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        criterion=criterion,
        epochs=1,
        mode="gcn",
    )

    assert len(losses) == 1
    assert losses[0] >= 0



def test_build_predictions_dataframe():
    y_true = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    y_pred = torch.tensor([[1.5, -1.0], [2.0, 5.0]])

    df = build_predictions_dataframe(y_true, y_pred)

    assert len(df) == 4
    assert set([
        "sample_index",
        "asset_index",
        "y_true",
        "y_pred",
        "error",
        "abs_error",
        "squared_error",
        "actual_direction",
        "predicted_direction",
        "direction_correct",
    ]).issubset(df.columns)
    assert df.loc[0, "sample_index"] == 0
    assert df.loc[0, "asset_index"] == 0
    assert df.loc[0, "error"] == pytest.approx(0.5)
    assert df.loc[0, "abs_error"] == pytest.approx(0.5)
    assert df.loc[0, "squared_error"] == pytest.approx(0.25)


def test_compute_regression_metrics():
    y_true = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    y_pred = torch.tensor([[1.0, -1.0], [2.0, 5.0]])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["mse"] == pytest.approx(0.75)
    assert metrics["rmse"] == pytest.approx(0.75 ** 0.5)
    assert metrics["mae"] == pytest.approx(0.75)
    assert metrics["directional_accuracy"] == pytest.approx(1.0)
    assert metrics["n_observations"] == 4
    assert len(metrics["per_asset_mae"]) == 2
