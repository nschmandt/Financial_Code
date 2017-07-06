import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lxml import html
import requests
from bs4 import BeautifulSoup
import pandas_datareader.data as web
import os


#this code collects the name of the stocks in the S&P 500

# page = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# soup=BeautifulSoup(page.content,'html.parser')
# table=soup.find('table')
# table_entries=table.findAll('td')
#
# stock_assets=[]
# for i in range(0, len(table_entries)):
#     stock_assets.append(table_entries[i].get_text())
#
# ticker_symbol=stock_assets[::8]
# company_name=stock_assets[1::8]
# company_sector=stock_assets[3::8]
# company_industry=stock_assets[4::8]
# stock_assets=pd.DataFrame([ticker_symbol, company_name, company_sector, company_industry], index=\
#     ['ticker', 'company', 'sector', 'industry'])
# stock_assets=stock_assets.transpose()
#
# os.chdir('C://Users//nts21//Documents//financial_models')
# stock_assets.to_csv('sp500_2017_list.csv')

#this code takes the S&P 500 output from the previous lines and harvests their stock values since 2010.

ticker_symbol=pd.read_csv('sp500_2017_list.csv', index_col=0)
start=pd.to_datetime('1-1-2010')
end=pd.to_datetime('7-3-2017')

temp=web.DataReader(ticker_symbol['ticker'][0], 'google', start, end)
sp500_close=temp['Close']
sp500_open=temp['Open']
sp500_low=temp['Low']
sp500_high=temp['High']
sp500_volume=temp['Volume']
sp500_fluct=temp['High']-temp['Low']
sp500_change=temp['Close']-temp['Open']
for i in ticker_symbol['ticker']:
    try:
        temp=web.DataReader(i, 'google', start, end)
        sp500_close[i]=temp['Close']
        sp500_open[i]=temp['Open']
        sp500_low[i]=temp['Low']
        sp500_high[i]=temp['High']
        sp500_volume[i]=temp['Volume']
        sp500_fluct[i]=temp['High']-temp['Low']
        sp500_change[i]=temp['Close']-temp['Open']
    except:
        print('Could not read data for ' +i)

os.chdir('C://Users//nts21//Documents//financial_models')
sp500_close.to_csv('sp500_close_2010_2017.csv')
sp500_open.to_csv('sp500_open_2010_2017.csv')
sp500_low.to_csv('sp500_low_2010_2017.csv')
sp500_high.to_csv('sp500_high_2010_2017.csv')
sp500_volume.to_csv('sp500_volume_2010_2017.csv')
sp500_fluct.to_csv('sp500_fluct_2010_2017.csv')
sp500_change.to_csv('sp500_change_2010_2017.csv')