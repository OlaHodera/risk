import numpy as np
from scipy import stats
from scipy.optimize import fsolve


def system(x, T, L, r, k, E, sigma_E, t=0):
    sigma, v = x
    d1 = (np.log(v / L) + ((r - k) + sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = (np.log(v / L) + ((r - k) - sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt(T - t))
    return v * stats.norm.cdf(d1) - L * np.exp(-r * T) * stats.norm.cdf(d2) - E, \
           sigma_E * E - stats.norm.cdf(d1) * sigma * v


def insolvency_probability(T, parameters, t=0):
    solution = fsolve(
        lambda x: system(x,
                         T,
                         parameters["L"],
                         parameters["r"],
                         parameters["k"],
                         parameters["E"],
                         parameters["sigma_E"]),
        x0=np.array([parameters["sigma_E"], parameters["market_cap"]]))
    sigma, v = solution[0], solution[1]
    print(solution)
    d2 = (np.log(v / parameters["L"]) + ((parameters["r"] - parameters["k"]) - sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt(T - t))
    return 1 - stats.norm.cdf(d2)
