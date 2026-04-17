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