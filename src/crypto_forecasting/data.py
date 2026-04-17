import pandas as pd
import os
import numpy as np
import torch

from torch.utils.data import IterableDataset

from .utils import *


from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class CryptoFeed(IterableDataset):
    def __init__(self, df, seq_len=5, technicals=None, evaluation=False, split_ratio=0.9, max_timesteps=100000):
        """
        Creates an iterable feed of crypto market states in chronological order.

        Args:
            df: pandas DataFrame containing crypto asset price data.
            seq_len: int, sliding window length for each sample.
            technicals: dict mapping feature names to functions that compute technical indicators.
            evaluation: bool, if True use the evaluation split; otherwise use the training split.
            split_ratio: float, proportion of rows used for training.
            max_timesteps: int, optional cap on the number of most recent timestamps to keep.
        """
        cached_features_path = DATA_DIR / "filtered_features.csv"
        cached_targets_path = DATA_DIR / "filtered_targets.csv"
        cached_log_returns_path = DATA_DIR / "filtered_log_returns.csv"

        if (
            cached_features_path.exists()
            and cached_targets_path.exists()
            and cached_log_returns_path.exists()
        ):
            self.features = pd.read_csv(cached_features_path).set_index("timestamp")
            self.targets = pd.read_csv(cached_targets_path).set_index("timestamp")
            self.log_returns = pd.read_csv(cached_log_returns_path).set_index("timestamp")
        else:
            unique_timestamps = sorted(df["timestamp"].unique())
            if len(unique_timestamps) > max_timesteps:
                accepted_timestamp_threshold = unique_timestamps[-max_timesteps]
                df = df[df["timestamp"] > accepted_timestamp_threshold]

            df = df.sort_values(["timestamp", "Asset_ID"]).copy()

            self.id_to_name = dict(zip(df["Asset_ID"], df["Asset_Name"]))
            self.data = [df[df["Asset_ID"] == i].copy() for i in sorted(df["Asset_ID"].unique())]

            self.targets = pd.concat(
                [tdf.set_index("timestamp")["Target"] for tdf in self.data],
                axis=1,
            )

            self.log_returns = pd.concat(
                [tdf.set_index("timestamp")["Close"] for tdf in self.data],
                axis=1,
            )
            self.log_returns = np.log(self.log_returns) - np.log(self.log_returns.shift(1))

            if technicals is not None:
                for tdf in self.data:
                    for feature_name, feature_fn in technicals.items():
                        tdf[feature_name] = feature_fn(tdf)

            for tdf in self.data:
                tdf.set_index("timestamp", inplace=True)

            for col in ["timestamp", "Asset_ID", "Asset_Name", "Target"]:
                for tdf in self.data:
                    if col in tdf.columns:
                        tdf.drop(columns=col, inplace=True)

            self.features = pd.concat(self.data, axis=1)

            self.features.to_csv(cached_features_path)
            self.targets.to_csv(cached_targets_path)
            self.log_returns.to_csv(cached_log_returns_path)

        self.features.ffill(0, inplace=True)
        self.targets.ffill(0, inplace=True)
        self.log_returns.ffill(0, inplace=True)

        split_idx = int(len(self.features) * split_ratio)

        if evaluation:
            self.features = self.features.iloc[split_idx:]
            self.targets = self.targets.iloc[split_idx:]
            self.log_returns = self.log_returns.iloc[split_idx:]
        else:
            self.features = self.features.iloc[:split_idx]
            self.targets = self.targets.iloc[:split_idx]
            self.log_returns = self.log_returns.iloc[:split_idx]

        self.seq_len = seq_len
        self.valid_dates = list(self.features.index)
    
    def __len__(self):
        return self.features.shape[0]
    
    def __iter__(self):
        """
        Yield one sample at a time for model training or evaluation.

        Yields:
            tuple:
                features: np.ndarray or torch.Tensor of shape (seq_len, total_feature_dim),
                    where total_feature_dim = n_assets * features_per_asset

                target: np.ndarray or torch.Tensor of shape (n_assets,),
                    one target value per asset

                adj: np.ndarray or torch.Tensor of shape (n_assets, n_assets),
                    adjacency matrix describing relationships between assets
        """
        
        for i in range(self.seq_len, len(self.valid_dates)):
            dates_idx = self.valid_dates[i-self.seq_len:i]
            features = self.features.loc[dates_idx].values
            target = self.targets.loc[dates_idx[-1]].values # target is target of end of window
            adj = self.log_returns.loc[dates_idx].corr().ffill(value=0).values # correlation matrix between previous seq_len target values
            yield features, target, adj


def get_crypto_dataset(seq_len=5, technicals=None, evaluation=False):
    file_path = DATA_DIR / "g-research-crypto-forecasting" / "train.csv"
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    file_path = DATA_DIR / "g-research-crypto-forecasting" / "asset_details.csv"
    asset_details = pd.read_csv(file_path)
    id_to_names = dict(zip(asset_details['Asset_ID'], asset_details['Asset_Name']))
    data['Asset_Name'] = [id_to_names[a] for a in data['Asset_ID']]
    data.ffill(method='ffill', inplace=True)
    data.ffill(value=0, inplace=True)

    dataset = CryptoFeed(data, seq_len, technicals, evaluation)
    return dataset

class MockCryptoFeed(IterableDataset):
    def __init__(self, seq_len=10, n_assets=14, features_per_asset=7, n_samples=100, seed=42):
        self.seq_len = seq_len
        self.n_assets = n_assets
        self.features_per_asset = features_per_asset
        self.n_samples = n_samples
        self.total_features = n_assets * features_per_asset
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        for _ in range(self.n_samples):
            features = self.rng.normal(size=(self.seq_len, self.total_features)).astype(np.float32)
            target = self.rng.normal(size=(self.n_assets,)).astype(np.float32)

            adj = self.rng.normal(size=(self.n_assets, self.n_assets)).astype(np.float32)
            adj = (adj + adj.T) / 2.0
            np.fill_diagonal(adj, 1.0)

            yield (
                torch.tensor(features),
                torch.tensor(target),
                torch.tensor(adj)
            )

def get_mock_crypto_dataset(seq_len=10, technicals=None, evaluation=False, n_samples=100):
    """
    Create a small synthetic dataset for smoke tests.

    Args:
        seq_len: int.
            Number of time steps per sample.

        technicals: dict or None.
            Mapping of technical indicator names to feature builders.
            Used only to determine feature dimensionality.

        evaluation: bool.
            Included for interface compatibility with the real dataset loader.

        n_samples: int.
            Number of synthetic samples to generate.

    Returns:
        MockCryptoFeed yielding tuples of:
            - features: shape (seq_len, total_feature_dim)
            - target: shape (14,)
            - adj: shape (14, 14)
    """
    features_per_asset = 7 + (0 if technicals is None else len(technicals))
    return MockCryptoFeed(
        seq_len=seq_len,
        n_assets=14,
        features_per_asset=features_per_asset,
        n_samples=n_samples
    )

    