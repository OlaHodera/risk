import numpy as np
import pandas as pd
import scipy.stats as stats
from tests import kupiec, christoffersen

#from projekt3.utils.download import parse_json

params = parse_json()
lam = params['lambda']
window = params['window']
small_window = window // 4


def calculate_var(df, method, name, alpha=95):
    df[f'var{alpha}_{name}'] = df.returns.rolling(window=window).agg(method)
    df[f'VaR{alpha}_{name}'] = df[f'var{alpha}_{name}'] * df.Close.shift(-1) + df.Close.shift(-1)
    return df


def exceeds_vector(df, name, alpha=95):
    df[f'exceeds{alpha}_{name}'] = np.where(df[f'returns'] > df[f'var{alpha}_{name}'], 1, 0)
    df.loc[df[f'var{alpha}_{name}'].isna(), f'exceeds{alpha}_{name}'] = np.nan
    df[f'procent_przekroczeń_{name}{alpha}'] = df[f'exceeds{alpha}_{name}'].\
                                                   rolling(window=small_window).sum() / small_window
    I = df[f'exceeds{alpha}_{name}'].dropna()  # wektor przekroczeń
    return df, I


def test_exceeds_vector(vector, alpha=.95):
    return pd.DataFrame({'empiryczne': [vector.mean(),
                                        vector.var(),
                                        stats.binomtest(int(vector.sum()), n=vector.size, p=1-alpha).pvalue,
                                        kupiec(vector, alpha),
                                        christoffersen(vector)],
                         'teoretyczne': [1 - alpha,
                                         alpha*(1-alpha),
                                         '-',
                                         '-',
                                         '-']},
                        index=['średnia', 'wariancja',
                               'binomtest p-wartość',
                               'Kupiec - p-wartość',
                               'christoffersen p-wartość'])
