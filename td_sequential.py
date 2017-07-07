import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lxml import html
import requests
from bs4 import BeautifulSoup
import pandas_datareader.data as web
import os
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import datetime as datetime

os.chdir('/home/nick/datasets/financial')
#ticker_symbol=pd.read_csv('sp500_2017_list.csv', index_col=0)
ticker_symbol=['MMM', 'WFC', 'AET']
start=pd.to_datetime('1-1-2010')
end=pd.to_datetime(datetime.date.today())


lose_occurrences = []
win_occurrences = []
lose_counter = 0
win_counter = 0
expected_return = []

for i in ticker_symbol:
    stock=pd.DataFrame()
    try:
        stock=web.DataReader(i, 'google', start, end)
    except:
        print('Unable to read data for ' + i)

    if not stock.empty:

        #get rid of any rows with null values at the beginning
        while len(stock[stock['High'].isnull() == True].index)>0:
            stock=stock.drop(stock.index[0], axis=0)

        if len(stock['Open']) > 500 and np.sum((np.sum(np.isnan(stock))))==0:

            stock['Change'] = (stock['Close'] - stock['Open']) / stock['Open']
            stock['Fluct'] = ((stock['High'] - stock['Low']) / stock['Open']) - stock['Change']

            #remove erroneous entries where the opening value is zero
            stock = stock.drop(stock[stock['Open'] == 0].index, axis=0)

            ten_day_average = list(np.zeros(50))
            ten_day_std = list(np.zeros(50))
            ten_day_average_fluct = list(np.zeros(50))
            forty_day_average = list(np.zeros(50))
            forty_day_std = list(np.zeros(50))
            forty_day_average_fluct = list(np.zeros(50))
            for j in range(50, len(stock['Change'])):
                ten_day_average.append(np.mean(stock['Close'][j - 10:j]))
                ten_day_average_fluct.append(np.mean(stock['Fluct'][j - 10:j]))
                forty_day_average.append(np.mean(stock['Close'][j - 40:j]))
                forty_day_average_fluct.append(np.mean(stock['Fluct'][j - 40:j]))

            stock['10_day_av'] = ten_day_average
            stock['40_day_av'] = forty_day_average
            stock['10_day_fluct_av'] = ten_day_average_fluct
            stock['40_day_fluct_av'] = forty_day_average_fluct

            for j in range(50, len(stock['Change'])):
                ten_day_std.append(np.std(stock['Close'][j-10:j])/stock['10_day_av'][j])
                forty_day_std.append(np.std(stock['Close'][j-40:j])/stock['40_day_av'][j])

            stock['10_day_std'] = ten_day_std
            stock['40_day_std'] = forty_day_std
            stock['10_day_change'] = (stock['Close']-stock['10_day_av'])/stock['10_day_av']
            stock['40_day_change'] = (stock['Close']-stock['40_day_av'])/stock['40_day_av']

            i=50
            while i<(len(stock['Close'])-50):
            #for i in range(50, len(stock['Close']) - 30):
                #this if statement contains the reverse td_sequential, the correct one is below.
                #if (stock['Close'][i + 4] < stock['Close'][i]) and (stock['Close'][i + 5] > stock['Close'][i + 1]) \
                        #and (stock['Close'][i + 13] < stock['Close'][i + 5]):
                if (stock['Close'][i + 4] > stock['Close'][i]) and (stock['Close'][i + 5] < stock['Close'][i + 1]) \
                        and (stock['Close'][i + 13] < stock['Close'][i + 5]):
                    if stock['Close'][i + 21] - stock['Close'][i - 13] < 0:
                        lose_occurrences.insert(lose_counter, stock['Volume'][i:i + 25])
                        lose_counter += 1
                    if stock['Close'][i + 21] - stock['Close'][i - 13] > 0:
                        win_occurrences.insert(win_counter, stock['Volume'][i:i + 25])
                        win_counter += 1
                    expected_return.append(stock['Close'][i + 22] - stock['Close'][i + 13])
                    i+=20
                else:
                    i+=1

        else:
            if len(stock['Open']) > 500:
                print('Insufficient data for ' + i +' stock to perfrom analysis.')
            elif np.sum((np.sum(np.isnan(stock))))!=0:
                print('NaN values present for ' + i + '.')

plt.plot(np.mean(lose_occurrences, axis=0))
plt.plot(np.mean(win_occurrences, axis=0))
plt.legend(['lose', 'win'])
print(win_counter)
print(lose_counter)
print(np.mean(expected_return))
# temp=['%.2f' % e for e in expected_return]
# print(temp)
#plt.show()

#print(type(win_occurrences))
#print(type(lose_occurrences[1]))

for i in range(0, len(lose_occurrences)):
    temp=list(lose_occurrences[i])
    lose_occurrences[i]=temp.append(0)
for i in range(0, len(win_occurrences)):
    win_occurrences[i]=list(win_occurrences[i]).append(0)

print(len(lose_occurrences[1]))
print(lose_occurrences[1])
total_occurrences=lose_occurrences+win_occurrences








