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

os.chdir('/home/nick/datasets/financial/charts/sp500')
file_list=os.listdir('/home/nick/datasets/financial/charts/sp500')

#measured_value='10_day_change_vol'
buy_date=14 #default 14
sell_date=21 #default 21 or 22 (using 21)
max_difference=[]
cols_to_use=['Change', 'Fluct', 'Vol_Fluct', '10_day_change_vol', '40_day_change_vol', '10_day_fluct_av', \
             '40_day_fluct_av', '10_day_std', '40_day_std', '10_day_std_vol', '40_day_std_vol', '10_day_change', \
             '40_day_change', '10_day_change_vol', '40_day_change_vol']
threshold_for_analysis=3
cols_to_use=['40_day_fluct_av']
measured_value='40_day_fluct_av'
threshold_cutoff=np.arange(.36, .41, .002)
threshold_return=[]
for mod_threshold in threshold_cutoff:

    lose_occurrences = []
    win_occurrences = []
    lose_counter = 0
    win_counter = 0
    expected_return = []
    mod_expected_return = []
    mod_win_counter = 0
    mod_lose_counter = 0

    for file_name in file_list:
        stock=pd.read_csv(file_name)

        i=50
        while i<(len(stock['Close'])-50):
        #for i in range(50, len(stock['Close']) - 30):
            #this if statement contains the reverse td_sequential, the correct one is below.
            #if (stock['Close'][i + 4] < stock['Close'][i]) and (stock['Close'][i + 5] > stock['Close'][i + 1]) \
                    #and (stock['Close'][i + 13] < stock['Close'][i + 5]):
            if (stock['Close'][i + 4] > stock['Close'][i]) and (stock['Close'][i + 5] < stock['Close'][i + 1]) \
                    and (stock['Close'][i + 13] < stock['Close'][i + 5]):
                if (stock['Close'][i + sell_date] - stock['Close'][i + buy_date]) < - threshold_for_analysis:
                    lose_occurrences.insert(lose_counter, stock[measured_value][i-15:i + 25])
                    lose_counter += 1
                if (stock['Close'][i + sell_date] - stock['Close'][i + buy_date]) > threshold_for_analysis:
                    win_occurrences.insert(win_counter, stock[measured_value][i-15:i + 25])
                    win_counter += 1

                #if np.mean(stock['Vol_Fluct'][i+3:i+10])>0: # on MMM expect=.304, w=17, l=9
                #if stock['Fluct'][i+7]<.0175:  #on MMM, expect=.007, w=17, l=14
                #if np.mean(stock['10_day_fluct_av'][i+7:i+10])>.02: #on ticker_symbol[::20], exp=.853, w=250, l=154, all data exp=.541
                #if np.mean(stock['10_day_std_vol'][i:i+6]) < mod_threshold: #on all data, only i+1, best was .4255, averaged over i+1 to i+6 is .4265
                #if np.mean(stock['40_day_fluct_av'][i+1:i+7])>mod_threshold: # >.02 on all data, exp=.42, w=7553, l=5492, >.021 all data, exp=.568, max .61 at .022
                #if stock['10_day_std_vol'][i]<.3275: # on all data, exp = .406, w=7673, l=5591
                #if np.mean(stock['40_day_std_vol'][i+10:i+13])<mod_threshold: # <.455 on all data, exp=.367, <.45 all data, exp=.369, best over all
                if np.mean(stock['40_day_std'][i:i+5])>mod_threshold:

                    mod_expected_return.append(stock['Close'][i + sell_date] - stock['Close'][i + buy_date])
                    if (stock['Close'][i + sell_date] - stock['Close'][i + buy_date]) < 0:
                        mod_lose_counter +=1
                    if (stock['Close'][i + sell_date] - stock['Close'][i + buy_date]) > 0:
                        mod_win_counter +=1
                expected_return.append(stock['Close'][i + sell_date] - stock['Close'][i + buy_date])
                i+=20
            else:
                i+=1



    #fig=plt.figure()
    #print(len(lose_occurrences))
    #plt.plot(np.mean(lose_occurrences, axis=0), label='lose')
    #plt.plot(np.mean(win_occurrences, axis=0), label='win')
    #plt.legend()
    #plt.title(measured_value)
    #save_name=measured_value + '.png'
    os.chdir('/home/nick/datasets/financial')
    #fig.savefig(save_name)
    os.chdir('/home/nick/datasets/financial/charts/sp500')
    #print(win_counter)
    #print(lose_counter)
    print('Threshold: '+str(mod_threshold))
    print(np.mean(expected_return))
    print(np.mean(mod_expected_return))
    threshold_return.append(np.mean(mod_expected_return))
    print('new win count ' + str(mod_win_counter))
    print('new lose count ' + str(mod_lose_counter))
os.chdir('/home/nick/datasets/financial')
fig=plt.figure()
plt.plot(threshold_cutoff, threshold_return)
plt.title('Return by 40 day std, i:i+5')
plt.xlabel('40 day std')
plt.ylabel('Expected return')
fig.savefig('40_day_std')
print(threshold_return)
    # temp=['%.2f' % e for e in expected_return]
    # print(temp)
    #plt.show()

#print('Confusion matrix for ' + i)
#print(pd.DataFrame(sklearn.metrics.confusion_matrix(test_Y, pred), columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +']))
#conf_matrix=sklearn.metrics.confusion_matrix(test_Y, pred)





