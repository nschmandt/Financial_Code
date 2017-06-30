import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lxml import html
import requests
from bs4 import BeautifulSoup
import pandas_datareader.data as web
import os


# this code collects the name of the stocks in the S&P 500

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
# os.chdir('/home/nick/datasets/financial')
# stock_assets.to_csv('sp500_2017_list.csv')

ticker_symbol=pd.read_csv('sp500_2017_list.csv')

temp=web.DataReader(i, 'google', start, end)
sp500_close=temp['Close']
for i in ticker_symbol:
    temp=web.DataReader(i, 'google', start, end)
    sp500_close[i]=temp['Close']
    sp500_open[i]=temp['Open']
    sp500_low[i]=temp['Low']
    sp500_high[i]=temp['High']
    sp500_volume[i]=temp['Volume']
    sp500_fluct[i]=temp['High']-temp['Low']
    sp500_change[i]=temp['Close']-temp['Open']

os.chdir('/home/nick/datasets/financial')
sp500_close.to_csv('sp500_close_2010_2017.csv')
sp500_close.to_csv('sp500_close_2010_2017.csv')
sp500_close.to_csv('sp500_close_2010_2017.csv')