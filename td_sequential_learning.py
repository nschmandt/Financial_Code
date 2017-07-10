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

start=pd.to_datetime('1-1-2010')
end=pd.to_datetime(datetime.date.today())

stock=web.DataReader('MMM', 'google', start, end)

stock['Change'] = (stock['Close'] - stock['Open']) / stock['Open']
stock['Fluct'] = ((stock['High'] - stock['Low']) / stock['Open']) - stock['Change']
stock['Vol_Fluct'] = (stock['Volume'] - np.mean(stock['Volume'])) / np.mean(stock['Volume'])

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
    ten_day_std.append(np.std(stock['Close'][j-10:j])/stock['10_day_av'][j])
    forty_day_std.append(np.std(stock['Close'][j-40:j])/stock['40_day_av'][j])
    ten_day_std_volume.append(np.std(stock['Volume'][j - 10:j]) / stock['10_day_av_vol'][j])
    forty_day_std_volume.append(np.std(stock['Volume'][j - 40:j]) / stock['40_day_av_vol'][j])

stock['10_day_std'] = ten_day_std
stock['40_day_std'] = forty_day_std
stock['10_day_std_vol'] = ten_day_std_volume
stock['40_day_std_vol'] = forty_day_std_volume
stock['10_day_change'] = (stock['Close']-stock['10_day_av'])/stock['10_day_av']
stock['40_day_change'] = (stock['Close']-stock['40_day_av'])/stock['40_day_av']
stock['10_day_change_vol'] = (stock['Volume'] - stock['10_day_av_vol']) / stock['10_day_av_vol']
stock['40_day_change_vol'] = (stock['Volume'] - stock['40_day_av_vol']) / stock['40_day_av_vol']



lose_occurrences = []
win_occurrences = []
lose_counter = 0
win_counter = 0
expected_return = []
mod_expected_return = []
mod_win_counter = 0
mod_lose_counter = 0
measured_value= 'Vol_Fluct'
buy_date = 14  # default 14
sell_date = 21  # default 21 or 22 (using 21)
max_difference = []
linreg_data = []

i = 50
while i < (len(stock['Close']) - 50):
    # for i in range(50, len(stock['Close']) - 30):
    # this if statement contains the reverse td_sequential, the correct one is below.
    # if (stock['Close'][i + 4] < stock['Close'][i]) and (stock['Close'][i + 5] > stock['Close'][i + 1]) \
    # and (stock['Close'][i + 13] < stock['Close'][i + 5]):
    add_values = []
    if (stock['Close'][i + 4] > stock['Close'][i]) and (stock['Close'][i + 5] < stock['Close'][i + 1]) \
            and (stock['Close'][i + 13] < stock['Close'][i + 5]):
        day_to_learn = i + 3
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
        linreg_data.append(add_values)
    else:
        i += 1


linreg_data=np.array(linreg_data)
#model=LinearRegression()
model=RandomForestClassifier(150, oob_score=True, n_jobs=-1)
train, test=sklearn.model_selection.train_test_split(linreg_data, test_size=.3, \
                                                     stratify=np.sign(linreg_data[:,10]))

fit=model.fit(train[:,0:10], np.sign(train[:, 10]))
pred=fit.predict(train[:,0:10])
print(np.sum(np.sign(pred)==np.sign(train[:,10]))/len(pred))
pred=fit.predict(test[:,0:10])
print(np.sum(np.sign(pred)==np.sign(test[:,10]))/len(pred))