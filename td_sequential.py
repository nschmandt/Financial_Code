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
ticker_symbol=ticker_symbol[::20]
#ticker_symbol=['MMM']
start=pd.to_datetime('1-1-2010')
end=pd.to_datetime(datetime.date.today())


lose_occurrences = []
win_occurrences = []
lose_counter = 0
win_counter = 0
expected_return = []
mod_expected_return = []
mod_win_counter = 0
mod_lose_counter = 0
#measured_value='10_day_change_vol'
buy_date=14 #default 14
sell_date=21 #default 21 or 22 (using 21)
max_difference=[]
cols_to_use=['Change', 'Fluct', 'Vol_Fluct', '10_day_change_vol', '40_day_change_vol', '10_day_fluct_av', \
             '40_day_fluct_av', '10_day_std', '40_day_std', '10_day_std_vol', '40_day_std_vol', '10_day_change', \
             '40_day_change', '10_day_change_vol', '40_day_change_vol']
for measured_value in cols_to_use:

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
                stock['Vol_Fluct'] = (stock['Volume'] - np.mean(stock['Volume'])) / np.mean(stock['Volume'])

                #remove erroneous entries where the opening value is zero
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

                i=50
                while i<(len(stock['Close'])-50):
                #for i in range(50, len(stock['Close']) - 30):
                    #this if statement contains the reverse td_sequential, the correct one is below.
                    #if (stock['Close'][i + 4] < stock['Close'][i]) and (stock['Close'][i + 5] > stock['Close'][i + 1]) \
                            #and (stock['Close'][i + 13] < stock['Close'][i + 5]):
                    if (stock['Close'][i + 4] > stock['Close'][i]) and (stock['Close'][i + 5] < stock['Close'][i + 1]) \
                            and (stock['Close'][i + 13] < stock['Close'][i + 5]):
                        if (stock['Close'][i + sell_date] - stock['Close'][i + buy_date]) < 0:
                            lose_occurrences.insert(lose_counter, stock[measured_value][i:i + 25])
                            lose_counter += 1
                        if (stock['Close'][i + sell_date] - stock['Close'][i + buy_date]) > 0:
                            win_occurrences.insert(win_counter, stock[measured_value][i:i + 25])
                            win_counter += 1




                        #if np.mean(stock['Vol_Fluct'][i+3:i+10])>0: # on MMM expect=.304, w=17, l=9
                        #if stock['Fluct'][i+7]<.0175:  #on MMM, expect=.007, w=17, l=14
                        #if np.mean(stock['10_day_fluct_av'][i+7:i+10])>.02: #on ticker_symbol[::20], exp=.853, w=250, l=154
                        if stock['10_day_std_vol'][i + 1] < .325:

                            mod_expected_return.append(stock['Close'][i + sell_date] - stock['Close'][i + buy_date])
                            if (stock['Close'][i + sell_date] - stock['Close'][i + buy_date]) < 0:
                                mod_lose_counter +=1
                            if (stock['Close'][i + sell_date] - stock['Close'][i + buy_date]) > 0:
                                mod_win_counter +=1
                        expected_return.append(stock['Close'][i + sell_date] - stock['Close'][i + buy_date])
                        i+=20
                    else:
                        i+=1

            else:
                if len(stock['Open']) > 500:
                    print('Insufficient data for ' + i +' stock to perfrom analysis.')
                elif np.sum((np.sum(np.isnan(stock))))!=0:
                    print('NaN values present for ' + i + '.')

    fig=plt.figure()
    plt.plot(np.mean(lose_occurrences, axis=0), label='lose')
    plt.plot(np.mean(win_occurrences, axis=0), label='win')
    plt.legend()
    plt.title(measured_value)
    save_name=measured_value + '.png'
    fig.savefig(save_name)
    print(win_counter)
    print(lose_counter)
    print(np.mean(expected_return))
    print(np.mean(mod_expected_return))
    print('new win count ' + str(mod_win_counter))
    print('new lose count ' + str(mod_lose_counter))
    # temp=['%.2f' % e for e in expected_return]
    # print(temp)
    #plt.show()

#print(type(win_occurrences))
#print(type(lose_occurrences[1]))

stratification=[]

for i in range(0, len(lose_occurrences)):
    temp=list(lose_occurrences[i])
    lose_occurrences[i]=temp + [0]
for i in range(0, len(win_occurrences)):
    temp=list(win_occurrences[i])
    win_occurrences[i]=temp+[1]

total_occurrences=np.array(lose_occurrences+win_occurrences)

model=LogisticRegression()
#model=RandomForestClassifier(150, oob_score=True, n_jobs=-1)
train, test=sklearn.model_selection.train_test_split(total_occurrences, test_size=.3, \
                                                     stratify=total_occurrences[0:len(total_occurrences),25])

fit=model.fit(train[0:len(train),0:10], train[0:len(train), 25])
pred=fit.predict(test[0:len(train),0:10])

#print('Confusion matrix for ' + i)
#print(pd.DataFrame(sklearn.metrics.confusion_matrix(test_Y, pred), columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +']))
#conf_matrix=sklearn.metrics.confusion_matrix(test_Y, pred)





