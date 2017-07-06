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

ticker_symbol=pd.read_csv('sp500_2017_list.csv', index_col=0)
#ticker_symbol=['AET']
start=pd.to_datetime('1-1-2010')
end=pd.to_datetime('7-3-2017')

analyzed_tickers=[]
change_accuracy=[]
sample_size=[]
correct_pos=[]
false_pos=[]

for i in ticker_symbol['ticker']:
    try:
        stock=web.DataReader(i, 'google', start, end)
        stock['Change']=stock['Close']-stock['Open']
        stock['Fluct']=stock['High']-stock['Low']

        while len(stock[stock['Open'].isnull() == True].index)>0:
            stock=stock.drop(stock.index[0], axis=0)


        if len(stock['Open'])>500:
            next_change=[]
            for j in range(0, len(stock['Change'])-1):
                if stock['Change'][j+1]>1:
                    next_change.append(1)
                else:
                    next_change.append(0)
            next_change.append(0)
            stock['next_change']=next_change

        #model=LogisticRegression()
            model=RandomForestClassifier(150, oob_score=True, n_jobs=-1)
            train, test=sklearn.model_selection.train_test_split(stock, test_size=.3, stratify=stock['next_change'])

            train_Y=train['next_change']
            cols_to_use=['Volume', 'Fluct', 'Change']
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
            print('Insufficient data for ' + i +' stock to perfrom analysis.')

    except:
        print('Unable to read data for ' + i)

total_data=[analyzed_tickers, change_accuracy, sample_size, correct_pos, false_pos]
total_data=pd.DataFrame(total_data)
total_data.to_csv('sp500_change_analysis_confmatrix.csv')