import numpy as np
import pandas as pd
from arch import arch_model
import scipy.stats as stats

from projekt3.utils.download import parse_json


params = parse_json()
lam = params['lambda']


def weighted_var(losses, alpha, statistic='quantile'):
    w1 = 1 / np.sum(lam ** np.arange(len(losses)))
    weights = w1 * lam ** np.arange(len(losses))[::-1]
    if statistic == 'expectile':
        return stats.expectile(losses, alpha=alpha, weights=weights)
    df = pd.DataFrame({'losses': losses, 'weights': weights}).reset_index(drop=True)
    df = df.sort_values(by='losses', ascending=True, ignore_index=True)
    df.weights = df.weights.cumsum()
    if statistic == 'quantile':
        return df.loc[df.weights.searchsorted(alpha)-1, 'losses']


def garch_var(losses, alpha, statistic='quantile'):
    losses = losses.dropna() * 100
    model = arch_model(losses, vol="GARCH", p=1, q=1, dist="t", mean='constant')
    res = model.fit()
    forecasts = res.forecast(horizon=1)
    variance = forecasts.variance.values[0, 0]
    mean = forecasts.mean.values[0, 0]
    residuals = (res.resid - res.params['mu']) / res.conditional_volatility
    if statistic == 'quantile':
        q = residuals.quantile(alpha)
        return (mean + np.sqrt(variance) * q) / 100
    elif statistic == 'expectile':
        q = stats.expectile(residuals, alpha)
        return (mean + np.sqrt(variance) * q) / 100
