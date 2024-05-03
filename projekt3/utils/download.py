import json
import datetime
import pandas as pd
import yfinance as yf

with open("data\\parametry.json") as file:
    params = json.load(file)
params = {**params['all'], **params[params['variable']]}
params['test'] = datetime.datetime.strptime(params['test'], '%Y-%m-%d')


def get_data():
    if params['file path'] == "":
        data = yf.download(tickers=params['ticker'],
                           start=params['start'],
                           end=params['end']).Close.to_frame()
    else:
        data = pd.read_csv(params['file path'])
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
    return data, params
