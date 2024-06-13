import numpy as np
from scipy import stats
from scipy.optimize import fsolve


def system(x, L, r, k, T, t, E, sigma_E):
    sigma, v = x
    d1 = (np.log(v/L) + ((r - k) + sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = (np.log(v/L) + ((r - k) - sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt(T - t))
    return v * stats.norm.cdf(d1) - L * np.exp(-r * T) * stats.norm.cdf(d2) - E,\
        sigma_E * E - stats.norm.cdf(d1) * sigma * v


def insolvency_probability(L, r, k, T, t, E, sigma_E):
    solution = fsolve(
        lambda x: system(x, L, r, k, T, t, E, sigma_E),
        x0=np.array([0, 0]))
    sigma, v = solution[0], solution[1]
    return stats.norm.cdf(
        (np.log(L/v) - ((r - k) - sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt(T - t)))
