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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import datetime as datetime


os.chdir('/home/nick/datasets/financial/charts/sp500')
file_list=os.listdir('/home/nick/datasets/financial/charts/sp500')

buy_date=13 #default 14
sell_date=21 #default 21 or 22 (using 21)

linear_pred_acc=[]
logistic_pred_acc=[]
randforest_pred_acc=[]

threshold_size=2

for day_to_analyze in range(0, 12):
    print(day_to_analyze)
    stock_data = []
    for file_name in file_list:
        stock=pd.read_csv(file_name)

        i = 50
        while i < (len(stock['Close']) - 50):
            # this if statement contains the reverse td_sequential, the correct one is below.
            # if (stock['Close'][i + 4] < stock['Close'][i]) and (stock['Close'][i + 5] > stock['Close'][i + 1]) \
            # and (stock['Close'][i + 13] < stock['Close'][i + 5]):
            add_values = []
            if (stock['Close'][i + 4] > stock['Close'][i]) and (stock['Close'][i + 5] < stock['Close'][i + 1]) \
                    and (stock['Close'][i + buy_date-1] < stock['Close'][i + 5]):
                day_to_learn = i + day_to_analyze
                add_values.append(stock['10_day_std'][day_to_learn])
                add_values.append(stock['40_day_std'][day_to_learn])
                add_values.append(stock['10_day_std_vol'][day_to_learn])
                add_values.append(stock['40_day_std_vol'][day_to_learn])
                add_values.append(stock['10_day_change'][day_to_learn])
                add_values.append(stock['40_day_change'][day_to_learn])
                add_values.append(stock['10_day_change_vol'][day_to_learn])
                add_values.append(stock['40_day_change_vol'][day_to_learn])
                add_values.append(stock['10_day_fluct_av'][day_to_learn])
                add_values.append(stock['40_day_fluct_av'][day_to_learn])
                add_values.append(stock['Close'][i + sell_date] - stock['Close'][i + buy_date])
                # if (stock['Close'][i + sell_date] - stock['Close'][i + buy_date]) < 0:
                # add_values.append(0)
                # if (stock['Close'][i + sell_date] - stock['Close'][i + buy_date]) > 0:
                # add_values.append(1)
                i += 20
                stock_data.append(add_values)
            else:
                i += 1

    print(len(stock_data))

    stock_features=np.array(stock_data)
    #model=LinearRegression()
    stock_features_full=stock_features
    print('Expected Return: ' +str(np.mean(stock_features_full[:, 10])))

    stock_features=stock_features[abs(stock_features[:,10])>threshold_size]

    print('Number of instances above threshold: ' + str(len(stock_features)))

    model=LinearRegression()
    train, test=sklearn.model_selection.train_test_split(stock_features, test_size=.3, \
                                                         stratify=np.sign(stock_features[:, 10]))

    fit=model.fit(train[:,0:10], train[:, 10])
    #pred=fit.predict(train[:,0:10])
    #print('Linear Regression Training Difference: ' + str(np.sum(abs(pred-train[:,10]))/len(pred)))
    pred=fit.predict(test[:,0:10])
    print('Linear Regression Test Difference: ' + str(np.sum(abs(pred-test[:,10]))/len(pred)))
    linear_pred_acc.append(np.sum(abs(pred-test[:,10]))/len(pred))

    stock_features[:, 10]=stock_features[:, 10]>0
    print(np.sum(stock_features[:, 10]))
    model=RandomForestClassifier(100, oob_score=True, n_jobs=-1)
    train, test=sklearn.model_selection.train_test_split(stock_features, test_size=.3, \
                                                         stratify=stock_features[:, 10])

    fit=model.fit(train[:,0:10], train[:, 10])
    #pred=fit.predict(train[:,0:10])
    #train_result.append(np.sum(pred == train[:, 10]) / len(pred))
    pred=fit.predict(test[:,0:10])
    randforest_pred_acc.append(np.sum(pred == test[:, 10]) / len(pred))
    print('Random Forest Accuracy: ' + str(np.sum(pred == test[:, 10]) / len(pred)))

    model=LogisticRegression()
    train, test=sklearn.model_selection.train_test_split(stock_features, test_size=.3, \
                                                         stratify=stock_features[:, 10])

    fit=model.fit(train[:,0:10], train[:, 10])
    pred=fit.predict(train[:,0:10])
    print('Logistic Regression Training Accuracy: ' + str(np.sum(pred==train[:,10])/len(pred)))
    pred=fit.predict(test[:,0:10])
    print('Logistic Regression Test Accuracy: ' + str(np.sum(pred==test[:,10])/len(pred)))
    logistic_pred_acc.append(np.sum(pred==test[:,10])/len(pred))

randforest_pred_acc=np.array(randforest_pred_acc)
logistic_pred_acc=np.array(logistic_pred_acc)

print('Best Random Forest: ' +str(max(randforest_pred_acc)))
print('Best Logistic Regression: ' + str(max(logistic_pred_acc)))