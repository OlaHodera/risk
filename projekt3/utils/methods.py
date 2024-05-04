import numpy as np
import pandas as pd
from arch import arch_model

from projekt3.utils.download import parse_json


params = parse_json()
lam = params['lambda']
alpha = params['alpha']


def weighted_var(losses):
    w1 = 1 / np.sum(lam ** np.arange(len(losses)))
    weights = w1 * lam ** np.arange(len(losses))
    df = pd.DataFrame({'losses': losses, 'weights': weights}).reset_index(drop=True)
    df = df.sort_values(by='losses', ascending=True, ignore_index=True)
    df.weights = df.weights.cumsum()
    return df.loc[df.weights.searchsorted(alpha)-1, 'losses']


def garch_var(losses):
    losses = losses.dropna() * 100
    model = arch_model(losses, vol="GARCH", p=1, q=1, dist="t", mean='constant')
    res = model.fit()
    forecasts = res.forecast(horizon=1)
    variance = forecasts.variance.values[0, 0]
    mean = forecasts.mean.values[0, 0]
    residuals = (res.resid - mean) / res.conditional_volatility
    q = residuals.quantile(alpha)
    return (mean + np.sqrt(variance) * q) / 100
