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
ticker_symbol=pd.read_csv('sp500_2017_list.csv', index_col=0)
#ticker_symbol=['AET']
start=pd.to_datetime('1-1-2010')
end=pd.to_datetime(datetime.date.today())

analyzed_tickers=[]
change_accuracy=[]
sample_size=[]
correct_pos=[]
false_pos=[]

for i in ticker_symbol['ticker']:
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

            #create classification variable, such as a change in the value of a stock.
            next_change=[]
            for j in range(0, len(stock['Change'])-1):
                if stock['Change'][j+1]<-.01:
                    next_change.append(1)
                else:
                    next_change.append(0)
            next_change.append(0)
            stock['next_change']=next_change

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

            print(i)

            #model=LogisticRegression()
            model=RandomForestClassifier(150, oob_score=True, n_jobs=-1)
            train, test=sklearn.model_selection.train_test_split(stock[50:], test_size=.3, stratify=stock['next_change'][50:])

            train_Y=train['next_change']
            cols_to_use=['Volume', 'Fluct', 'Change', '10_day_std', '10_day_change', '10_day_fluct_av', \
                         '40_day_std', '40_day_change', '40_day_fluct_av']
            train_X=train[cols_to_use]
            fit=model.fit(train_X, train_Y)
            test_Y=test['next_change']
            test_X=test[cols_to_use]
            pred=fit.predict(test_X)

            #print('Confusion matrix for ' + i)
            #print(pd.DataFrame(sklearn.metrics.confusion_matrix(test_Y, pred), columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +']))
            conf_matrix=sklearn.metrics.confusion_matrix(test_Y, pred)

            analyzed_tickers.append(i)
            change_accuracy.append(np.mean(pred==test_Y))
            sample_size.append(np.sum(conf_matrix))
            correct_pos.append(conf_matrix[1,1])
            false_pos.append(conf_matrix[0,1])

        else:
            if len(stock['Open']) > 500:
                print('Insufficient data for ' + i +' stock to perfrom analysis.')
            elif np.sum((np.sum(np.isnan(stock))))!=0:
                print('NaN values present for ' + i + '.')

total_data=[analyzed_tickers, change_accuracy, sample_size, correct_pos, false_pos]
total_data=pd.DataFrame(total_data)
total_data.to_csv('sp500_neg_change_analysis_confmatrix.csv')
