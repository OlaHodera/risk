import scipy.stats as stats
import numpy as np


def var(L, weights, alpha, days=1):
    mean = L.mean()
    cov = L.cov()
    portfolio_mean = np.dot(mean, weights)
    portfolio_variance = np.dot(weights ** 2, np.diag(cov) ** 2) + 2 * (weights * (weights * cov).T).sum().sum()
    return np.sqrt(portfolio_variance * days) * stats.norm.ppf(alpha) + portfolio_mean * days