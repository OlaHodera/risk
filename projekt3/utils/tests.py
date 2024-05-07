import numpy as np
import scipy.stats as stats


# https://www.mathworks.com/help/risk/overview-of-var-backtesting.html


def kupiec(exceeds_vector, var_level=.95, alpha=.05):
    p, x, N = 1 - var_level, exceeds_vector.sum(), len(exceeds_vector)
    numerator = (1 - p) ** (N - x) * p ** x
    denominator = (1 - x / N) ** (N - x) * (x / N) ** x
    LR = -2 * np.log(numerator / denominator)
    return 1 - stats.chi2.cdf(LR, df=1)
