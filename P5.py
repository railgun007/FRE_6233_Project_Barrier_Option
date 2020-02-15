from P1 import *
from pandas_datareader import data
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Get data
start_date = '2017-12-03'
end_date = '2019-12-02'
all_data=yf.download('FXI', start_date, end_date)
start_date = '2017-12-03'
end_date = '2018-12-02'
close=all_data["Adj Close"].loc[start_date:end_date]
X=np.min(close)-2
ret=np.diff(np.log(close))
#252 is 252 trading days
sigma=np.std(ret)*np.sqrt(252)
start_date = '2018-12-03'
end_date = '2019-12-03'
test=all_data["Close"].loc[start_date:end_date]
K=test[0]
r=0.02
T=1
Rebate=10

# Below are the code to calculate pnl and draw price comparison pictures
def Pnl_doc(test,K,r,sigma,T,X,Rebate):
    d1=delta_of_doc(K, K, r, sigma, T, X, Rebate)
    PnL=[0]*(len(test)-1)
    delta_list=[0]*(len(test)-1)
    set_of_doc=[]
    set_of_vanilla_call=[]
    for i in range(1,len(test)):
        sell=max(Doc_with_rebate(test[i], K, r, sigma, T, X, Rebate),0)- max(Doc_with_rebate(test[i-1],K,r,sigma,T,X,Rebate),0)
        set_of_doc.append(Doc_with_rebate(test[i], K, r, sigma, T, X, Rebate))
        set_of_vanilla_call.append(Call(test[i], K, r, sigma, T))
        buy=(test[i]-test[i-1])*d1
        PnL[i-1]=1000*(sell-buy)
        delta_list[i-1]=d1
        d1=delta_of_doc(test[i], K, r, sigma, T, X, Rebate)
    plt.plot(PnL)
    plt.title("Down-and-out call hedging Profit and Loss")
    plt.xlabel("Date")
    plt.ylabel("PnL")
    plt.show()
    sumvalue=np.sum(PnL)
    plt.xlabel("Date")
    plt.ylabel("option price ")
    plt.plot(set_of_doc,color='red', label='Down-and-out Call with rebate')
    plt.plot(set_of_vanilla_call,color='blue', label='Vanilla call')
    plt.legend()
    plt.show()
    print(sumvalue)
    return sumvalue

def Pnl_dip(test,K,r,sigma,T,X,Rebate):
    d1=delta_of_dip(K, K, r, sigma, T, X, Rebate)
    PnL=[0]*(len(test)-1)
    delta_list=[0]*(len(test)-1)
    set_of_vanilla_put=[]
    set_of_dip=[]
    for i in range(1,len(test)):
        sell=max(Dip_with_rebate(test[i], K, r, sigma, T, X, Rebate),0)- max(Dip_with_rebate(test[i-1],K,r,sigma,T,X,Rebate),0)
        set_of_dip.append(Dip_with_rebate(test[i], K, r, sigma, T, X, Rebate))
        set_of_vanilla_put.append(Put(test[i], K, r, sigma, T))
        buy=(test[i]-test[i-1])*d1
        PnL[i-1]=1000*(sell-buy)
        delta_list[i-1]=d1
        d1=delta_of_dip(test[i], K, r, sigma, T, X, Rebate)
    plt.plot(PnL)
    plt.title("Down-and-in put hedging Profit and Loss")
    plt.xlabel("Date")
    plt.ylabel("PnL")
    plt.show()
    sumvalue=np.sum(PnL)
    plt.xlabel("Date")
    plt.ylabel("option price ")
    plt.plot(set_of_dip, color='red', label='Down-and-in put with rebate')
    plt.plot(set_of_vanilla_put, color='blue', label='Vanilla put option price')
    plt.legend()
    plt.show()
    print(sumvalue)
    return sumvalue

#Question 5
Pnl_doc(test,K,r,sigma,T,X,Rebate)
Pnl_dip(test,K,r,sigma,T,X,Rebate)
