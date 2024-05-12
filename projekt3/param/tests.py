import numpy as np
import scipy.stats as stats


def kupiec(exceeds_vector, var_level=.95):
    p, x, N = 1 - var_level, exceeds_vector.sum(), len(exceeds_vector)
    numerator = (1 - p) ** (N - x) * p ** x
    denominator = (1 - x / N) ** (N - x) * (x / N) ** x
    LR = -2 * np.log(numerator / denominator)
    return 1 - stats.chi2.cdf(LR, df=1)


def christoffersen(exceedance):
    pairs = list(zip(exceedance, exceedance[1:]))
    n = len(exceedance)
    i_0, i_1 = n - exceedance[1: -1].sum() - 2, exceedance[1: -1].sum()
    i_00, i_01, i_10, i_11 = pairs.count((0, 0)), pairs.count((0, 1)), pairs.count((1, 0)), pairs.count((1, 1))
    pi_01 = i_01 / (i_00 + i_01)
    pi_11 = i_11 / (i_10 + i_11)
    p = i_1 / n
    nested = (1 - p) ** i_0 * p ** i_1
    general = (1 - pi_01) ** i_00 * pi_01 ** i_01 * (1 - pi_11) ** i_10 * pi_11 ** i_11
    LR = -2 * np.log(nested / general)
    return 1 - stats.chi2.cdf(LR, df=1)


def scoring_expectile(returns, evars, alpha=.95):
    zeros = np.zeros(len(returns))
    S = alpha * np.maximum(returns - evars, zeros) ** 2\
        + (1 - alpha) * np.maximum(evars - returns, zeros) ** 2
    return S[~np.isnan(S)]


def scoring_quantile(returns, var, alpha=.95):
    zeros = np.zeros(len(returns))
    S = alpha * np.maximum(returns - var, zeros) + (1 - alpha) * np.maximum(var - returns, zeros)
    return S[~np.isnan(S)]
