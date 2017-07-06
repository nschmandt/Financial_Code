expected_return=0
for i in range(0, len(stock['Close'])-30):
    if (stock['Close'][i+4]>stock['Close'][i]) and (stock['Close'][i+5]<stock['Close'][i+1]) and (stock['Close'][i+13]<stock['Close'][i+5]):
        plt.plot(1, stock['Close'][i], 26, stock['Close'][i+26])
plt.show()