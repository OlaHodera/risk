import json
import pandas as pd
import yfinance as yf


def parse_json():
    with open("data\\parametry.json") as file:
        params = json.load(file)
    params = {**params['all'], **params[params['variable']]}
    return params


def get_data(yahoo=False):
    params = parse_json()
    if params['file path'] == "" or yahoo:
        data = yf.download(tickers=params['ticker'],
                           start=params['start'],
                           end=params['end']).Close.to_frame()
        data.to_csv(f'data\\{params["name"]}.csv')
    else:
        data = pd.read_csv(params['file path'], index_col='Date', header=0)
        data = data.loc[params['start']:params['end'], ]
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
    return data, params
