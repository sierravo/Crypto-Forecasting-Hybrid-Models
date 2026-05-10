"""
Various functions that may be helpful in the preprocessing steps of pipeline such as technical indicator calculations
Source/links for technical indicators:
    - 
"""

import numpy as np
import pandas as pd


def EMA(df, window, price_col="VWAP"):
    """
    Calculates exponential moving average of VWAP using the standard EMA convention.
    """
    return df[price_col].ewm(span=window, adjust=False).mean()


def SMA(df, window, price_col="VWAP"):
    """
    Calculates simple moving average of VWAP over a rolling window.
    """
    return df[price_col].rolling(window=window, min_periods=1).mean()


def EMA_5(df):
    return EMA(df, 5)


def EMA_20(df):
    return EMA(df, 20)


def EMA_50(df):
    return EMA(df, 50)


def SMA_5(df):
    return SMA(df, 5)


def SMA_20(df):
    return SMA(df, 20)


def SMA_50(df):
    return SMA(df, 50)
    

def RSI(df, window=14, price_col="VWAP"):
    """
    Calculates Relative Strength Index (RSI) using a rolling window.
    """
    delta = df[price_col].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(avg_loss == 0, 100)
    return rsi.fillna(0)


def BollingerBands(df, window=20, price_col="VWAP"):
    """
    Calculates bollinger bands which is basically just 20 day lookback volatility
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands
    """
    return df[price_col].rolling(window=window, min_periods=1).std()


def StochasticOscillator(df, window=14, price_col="VWAP"):
    """
    Calculates %K stochastic oscillator from VWAP.
    """
    low = df[price_col].rolling(window=window, min_periods=window).min()
    high = df[price_col].rolling(window=window, min_periods=window).max()
    denom = high - low
    pct_k = 100 * (df[price_col] - low) / denom
    return pct_k.mask(denom == 0, 0.0)

def compute_regression_metrics(y_true, y_pred, eps=1e-12):
    """
    Compute decision-oriented regression metrics for multi-asset forecasts.

    Args:
        y_true: array-like or torch.Tensor of shape (n_samples, n_assets) or compatible.
        y_pred: array-like or torch.Tensor with the same shape as y_true.
        eps: small float used to avoid divide-by-zero in optional calculations.

    Returns:
        dict with aggregate metrics:
            - mse
            - rmse
            - mae
            - directional_accuracy
            - n_observations

        If the input is 2D, also includes per-asset metrics:
            - per_asset_mse
            - per_asset_rmse
            - per_asset_mae
            - per_asset_directional_accuracy
    """
    try:
        import torch
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
    except ImportError:
        pass

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_true.shape} and {y_pred.shape}.")

    errors = y_pred - y_true
    squared_errors = errors ** 2
    absolute_errors = np.abs(errors)

    metrics = {
        "mse": float(np.mean(squared_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "mae": float(np.mean(absolute_errors)),
        "directional_accuracy": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
        "n_observations": int(y_true.size),
    }

    if y_true.ndim == 2:
        per_asset_mse = np.mean(squared_errors, axis=0)
        per_asset_rmse = np.sqrt(per_asset_mse)
        per_asset_mae = np.mean(absolute_errors, axis=0)
        per_asset_directional_accuracy = np.mean(np.sign(y_pred) == np.sign(y_true), axis=0)

        metrics.update({
            "per_asset_mse": per_asset_mse.tolist(),
            "per_asset_rmse": per_asset_rmse.tolist(),
            "per_asset_mae": per_asset_mae.tolist(),
            "per_asset_directional_accuracy": per_asset_directional_accuracy.tolist(),
        })

    return metrics



def set_random_seed(seed=42):
    """
    Set random seeds for reproducible training and evaluation runs.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_technical_registry():
    """
    Return a safe registry mapping config strings to technical indicator functions.

    This replaces eval()-based loading so JSON config values must match known
    indicator names.
    """
    return {
        "EMA_5": EMA_5,
        "EMA_20": EMA_20,
        "EMA_50": EMA_50,
        "SMA_5": SMA_5,
        "SMA_20": SMA_20,
        "SMA_50": SMA_50,
        "RSI": RSI,
        "BollingerBands": BollingerBands,
        "StochasticOscillator": StochasticOscillator,
    }


def resolve_technicals_config(config):
    """
    Convert a JSON technical-indicator config into callable functions.

    Args:
        config: dict mapping output feature names to registry keys.

    Returns:
        dict mapping output feature names to callable functions.
    """
    registry = get_technical_registry()
    resolved = {}

    for feature_name, function_name in config.items():
        if function_name not in registry:
            valid = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown technical indicator '{function_name}' for feature '{feature_name}'. "
                f"Valid options are: {valid}"
            )
        resolved[feature_name] = registry[function_name]

    return resolved
