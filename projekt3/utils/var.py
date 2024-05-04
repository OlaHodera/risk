import numpy as np


def calculate_var(df, method, name):
    df[f'var95_{name}'] = df.returns.rolling(window=125).agg(method)
    df[f'VaR95_{name}'] = df[f'var95_{name}'] * df.Close.shift(-1) + df.Close.shift(-1)
    return df


def exceeds_vector(df, name):
    df[f'exceeds_{name}'] = np.where(df[f'returns'] > df[f'var95_{name}'], 1, 0)
    df.loc[df[f'var95_{name}'].isna(), f'exceeds_{name}'] = np.nan
    df[f'procent_przekroczeń_{name}'] = df[f'exceeds_{name}'].rolling(window=50).sum() / 50
    I = df.dropna()[f'exceeds_{name}']  # wektor przekroczeń
    return df, I
