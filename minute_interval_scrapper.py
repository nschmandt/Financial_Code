import urllib.request
import pandas as pd
import os
import numpy as np

#this program scrapes all available data on the minute trading intervals from the sp500 list compiled earlier. All
#the data google will give is taken, in groups of about 12 days.
#data is processed to ensure it is completely 1-390 datasets with no more than 4 omissions. If it fails to meet that
#criteria, it is thrown out.

#load list of sp500 stocks
os.chdir('/home/nick/datasets/financial')
stocks=pd.read_csv('sp500_2017_list.csv', index_col=0)
#stocks=['AAPL', 'IBM', 'MSFT']

previous_stock_length=0
stock_list=[]
for stock_symbol in stocks['ticker']:
    stock_data = urllib.request.urlopen('https://www.google.com/finance/getprices?i=60&p=15d&f=d,o,h,l,c,v&df=cpct&q='+stock_symbol).readlines()
    print(stock_symbol)
    previous_stock_length = len(stock_list)

    for i in stock_data:
        i=i.decode('utf-8')
        temp=i.split(',')
        if len(temp)==6:
            if temp[5][-1:]=='\n':
                temp[5]=temp[5][:-1]
            try:
                int(temp[0])
                stock_list.append(temp)
            except:
                continue

    # we only want complete days
    temp_value = 0
    #this code cuts off any part that occurs after the last 390 or 389, so that the code ends after the last complete day.
    for i in range(0, len(stock_list)):
        if int(stock_list[i][0]) == 390 or int(stock_list[i][0]) == 389:
            temp_value = i
    if len(stock_list) - temp_value - 1 > 0:
        del stock_list[-(len(stock_list) - temp_value - 1):]
    list_incomplete = True
    index = 0
    print(stock_symbol)
    #this loop eliminates incomplete days in the middle or days with more than 4 minutes of missing data
    while list_incomplete:
        print('Stock List Current Length:' + str(len(stock_list)))
        print('index is now:' + str(index))
        if int(stock_list[index][0]) == 1 or int(stock_list[index][0]) == 2:
            for i in range(1, np.min([len(stock_list) - index, 391])):
                if (int(stock_list[index + i][0]) == 1 or int(stock_list[index + i][0]) == 2) and i > 5:
                    if i > 386:
                        index += i
                        break
                    else:
                        print('i index:' + str(i))
                        print(stock_list[index + i][0])
                        print(index)
                        print(index + i)
                        del stock_list[index:index + i]
                        break
                if (int(stock_list[index + i][0]) % 390) != i + 1:
                    if (int(stock_list[index + i][0]) % 390 - i - 1) < 5:
                        continue
                    else:
                        print('i index:' + str(i))
                        print(stock_list[index + i][0])
                        print(index)
                        print(index + i)
                        del stock_list[index:index + i]
                        break
            if i == np.min([len(stock_list) - index, 391]) - 1:
                index += np.min([len(stock_list) - index, 391]) - 1
                print(index)
        else:
            del stock_list[index]
        if index > len(stock_list) - 387:
            del stock_list[index:]
            list_incomplete = False
    print(len(stock_list))

#save the data as a pandas dataframe
stock_df=pd.DataFrame(stock_list, columns=['Time', 'Close', 'High', 'Low', 'Open', 'Volume'])
stock_df.to_csv('sp500_2017_minute_data.csv')
print(len(stock_list))