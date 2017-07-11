import numpy as np
import pandas as pd
from lxml import html
import requests
from bs4 import BeautifulSoup
import pandas_datareader.data as web
import os
import datetime as datetime



os.chdir('/home/nick/datasets/financial')
ticker_symbol=pd.read_csv('sp500_2017_list.csv', index_col=0)
start=pd.to_datetime('1-1-2010')
end=pd.to_datetime(datetime.date.today())

os.chdir('/home/nick/datasets/financial/charts/sp500')
for i in ticker_symbol['ticker']:
    stock=pd.DataFrame()
    try:
        stock=web.DataReader(i, 'google', start, end)
    except:
        print('Could not read data for '+i)

    if not stock.empty:

        # get rid of any rows with null values at the beginning
        while len(stock[stock['High'].isnull() == True].index) > 0:
            stock = stock.drop(stock.index[0], axis=0)

        if len(stock['Open']) > 500 and np.sum((np.sum(np.isnan(stock)))) == 0:

            stock['Change'] = (stock['Close'] - stock['Open']) / stock['Open']
            stock['Fluct'] = ((stock['High'] - stock['Low']) / stock['Open']) - stock['Change']
            stock['Vol_Fluct'] = (stock['Volume'] - np.mean(stock['Volume'])) / np.mean(stock['Volume'])

            # remove erroneous entries where the opening value is zero
            stock = stock.drop(stock[stock['Open'] == 0].index, axis=0)

            ten_day_average = list(np.zeros(50))
            ten_day_average_volume = list(np.zeros(50))
            ten_day_std = list(np.zeros(50))
            ten_day_std_volume = list(np.zeros(50))
            ten_day_average_fluct = list(np.zeros(50))
            forty_day_average = list(np.zeros(50))
            forty_day_average_volume = list(np.zeros(50))
            forty_day_std = list(np.zeros(50))
            forty_day_std_volume = list(np.zeros(50))
            forty_day_average_fluct = list(np.zeros(50))
            for j in range(50, len(stock['Change'])):
                ten_day_average.append(np.mean(stock['Close'][j - 10:j]))
                ten_day_average_fluct.append(np.mean(stock['Fluct'][j - 10:j]))
                forty_day_average.append(np.mean(stock['Close'][j - 40:j]))
                forty_day_average_fluct.append(np.mean(stock['Fluct'][j - 40:j]))
                ten_day_average_volume.append(np.mean(stock['Volume'][j - 10:j]))
                forty_day_average_volume.append(np.mean(stock['Volume'][j - 40:j]))

            stock['10_day_av'] = ten_day_average
            stock['40_day_av'] = forty_day_average
            stock['10_day_fluct_av'] = ten_day_average_fluct
            stock['40_day_fluct_av'] = forty_day_average_fluct
            stock['10_day_av_vol'] = ten_day_average_volume
            stock['40_day_av_vol'] = forty_day_average_volume

            for j in range(50, len(stock['Change'])):
                ten_day_std.append(np.std(stock['Close'][j - 10:j]) / stock['10_day_av'][j])
                forty_day_std.append(np.std(stock['Close'][j - 40:j]) / stock['40_day_av'][j])
                ten_day_std_volume.append(np.std(stock['Volume'][j - 10:j]) / stock['10_day_av_vol'][j])
                forty_day_std_volume.append(np.std(stock['Volume'][j - 40:j]) / stock['40_day_av_vol'][j])

            stock['10_day_std'] = ten_day_std
            stock['40_day_std'] = forty_day_std
            stock['10_day_std_vol'] = ten_day_std_volume
            stock['40_day_std_vol'] = forty_day_std_volume
            stock['10_day_change'] = (stock['Close'] - stock['10_day_av']) / stock['10_day_av']
            stock['40_day_change'] = (stock['Close'] - stock['40_day_av']) / stock['40_day_av']
            stock['10_day_change_vol'] = (stock['Volume'] - stock['10_day_av_vol']) / stock['10_day_av_vol']
            stock['40_day_change_vol'] = (stock['Volume'] - stock['40_day_av_vol']) / stock['40_day_av_vol']

            stock.to_csv(i + '.csv')

        else:
            if len(stock['Open']) < 500:
                print('Insufficient data for ' + i + ' stock to perfrom analysis.')
            elif np.sum((np.sum(np.isnan(stock)))) != 0:
                print('NaN values present for ' + i + '.')
