from scipy.stats import norm
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math



# Below are the code to calculate the value of options at time 0
def Call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T) / sigma * np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
    part1 = S * norm.cdf(d1)
    part2 = K * np.exp(-r * T) * norm.cdf(d2)
    return part1 - part2


def Put(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T) / sigma * np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
    part1 = norm.cdf(-d2) * K * np.exp(-r * T)
    part2 = norm.cdf(-d1) * S
    return part1 - part2


def Down_and_out_call(S, K, r, sigma, T, X):
    if S > X and K > X :
        return Call(S, K, r, sigma, T) - pow(X / S, 2 * r / (sigma * sigma) - 1) * Call(pow(X, 2) / S, K, r, sigma, T)
    elif S > X and K < X :
        lmd = (r + sigma * sigma / 2) / (sigma * sigma)
        x1 = np.log(S / X) / (sigma * np.sqrt(T)) + lmd * sigma * np.sqrt(T)
        y = np.log(X * X / (S * K)) / (sigma * np.sqrt(T)) + lmd * sigma * np.sqrt(T)
        y1 = np.log(X / S) / (sigma * np.sqrt(T)) + lmd * sigma * np.sqrt(T)
        part1 = S * norm.cdf(x1)
        part2 = K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
        part3 = S * pow(X / S, 2 * lmd) * norm.cdf(y1)
        part41 = K * np.exp(-r * T) * pow(X / S, 2 * lmd - 2)
        part42 = norm.cdf(y1 - sigma * np.sqrt(T))
        return part1-part2-part3+part41*part42
    else:
        return 0


def Down_and_in_put(S, K, r, sigma, T, X):
    lmd = (r + sigma * sigma / 2) / (sigma * sigma)
    x1 = np.log(S / X) / (sigma * np.sqrt(T)) + lmd * sigma * np.sqrt(T)
    y = np.log(X * X / (S * K)) / (sigma * np.sqrt(T)) + lmd * sigma * np.sqrt(T)
    y1 = np.log(X / S) / (sigma * np.sqrt(T)) + lmd * sigma * np.sqrt(T)
    part1 = -S * norm.cdf(-x1)
    part2 = K * np.exp(-r * T) * norm.cdf(-x1 + sigma * np.sqrt(T))
    part3 = S * pow(X / S, 2 * lmd) * (norm.cdf(y) - norm.cdf(y1))
    part41 = K * np.exp(-r * T) * pow(X / S, 2 * lmd - 2)
    part42 = norm.cdf(y - sigma * np.sqrt(T)) - norm.cdf(y1 - sigma * np.sqrt(T))
    if S > X and K > X: return part1 + part2 + part3 - part41 * part42
    else: return Put(S, K, r, sigma, T)


def European_digital_put(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * norm.cdf(-d2)


def European_digital_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * norm.cdf(d2)


def Put_eur_digital_rebate(S, K, r, sigma, T, X):
    lmd = (r + sigma * sigma / 2) / (sigma * sigma)
    part1 = European_digital_put(S, X, r, sigma, T)
    part2 = pow(S / X, 2 - 2 * lmd) * European_digital_call(X * X / S, X, r, sigma, T)
    if S > X:
        return part1 + part2
    else:
        return np.exp(-r*T)



def Anti_put_eur_digital_rebate(S, K, r, sigma, T, X):
    if S > X: return np.exp(-r * T) - Put_eur_digital_rebate(S, K, r, sigma, T, X)
    else: return 0


def Doc_with_rebate(S, K, r, sigma, T, X, Rebate):
    return Down_and_out_call(S, K, r, sigma, T, X) + Rebate * Put_eur_digital_rebate(S, K, r, sigma, T, X)


def Dip_with_rebate(S, K, r, sigma, T, X, Rebate):
    return Down_and_in_put(S, K, r, sigma, T, X) + Rebate * Anti_put_eur_digital_rebate(S, K, r, sigma, T, X)


# Below are the code to implement Monte Carlo simulation; dip stands for down-and-in-put
def Monte_dip(X, K, S, T, sigma, r, Rebate):
    np.random.seed(1234)
    result = 0
    m = 365
    number = 15000
    for i in range(0, number):
        indicator = 0
        Snew = S
        for j in range(0, m):
            Snew = Snew * math.exp(
                                   (r - sigma * sigma / 2) * (T / m) + sigma * math.sqrt(T / m) * np.random.normal(loc=0, scale=1, size=1))
            if Snew < X: indicator = 1
        if indicator == 1: result = result + max(K - Snew, 0)
        if indicator == 0: result = result + Rebate
    result = result / number
    result = result * math.exp(-r * T)
    return result


def Monte_doc(X, K, S, T, sigma, r, Rebate):
    np.random.seed(1234)
    result = 0
    m = 365
    number = 15000
    for i in range(0, number):
        indicator = 1
        Snew = S
        for j in range(0, m):
            Snew = Snew * math.exp(
                                   (r - sigma * sigma / 2) * (T / m) + sigma * math.sqrt(T / m) * np.random.normal(loc=0, scale=1, size=1))
            if Snew < X: indicator = 0
        if indicator == 1: result = result + max(Snew - K, 0)
        if indicator == 0: result = result + Rebate
    result = result / number
    result = result * math.exp(-r * T)
    return result


def Monte_rebate(S, K, r, sigma, T, X):
    np.random.seed(1234)
    result = 0
    m = 365
    number = 10000
    for i in range(0, number):
        indicator = 0
        Snew = S
        for j in range(0, m):
            Snew = Snew * math.exp(
                                   (r - sigma * sigma / 2) * (T / m) + sigma * math.sqrt(T / m) * np.random.normal(loc=0, scale=1, size=1))
            if Snew < X:
                result = result + 1
                break
        # if indicator == 0: result = result + 0
    result = float(result) / number
    result = result * math.exp(-r * T)
    return result


# Below are the code to calculate delta; doc stands for down-and-out-call
def delta_of_doc(S,K,r,sigma,T,X,Rebate):
    delta=[0]*100
    epsilon=0.1
    for i, j in enumerate(np.linspace(0,epsilon,100)):
        larger=Doc_with_rebate(S+j,K,r,sigma,T,X,Rebate)
        smaller=Doc_with_rebate(S-j,K,r,sigma,T,X,Rebate)
        value=(larger-smaller)/(2*epsilon)
        delta[i]=value
    return np.mean(delta)


def delta_of_dip(S,K,r,sigma,T,X,Rebate):
    delta=[0]*100
    epsilon=0.1
    for i, j in enumerate(np.linspace(0,epsilon,100)):
        larger=Dip_with_rebate(S+j,K,r,sigma,T,X,Rebate)
        smaller=Dip_with_rebate(S-j,K,r,sigma,T,X,Rebate)
        value=(larger-smaller)/(2*epsilon)
        delta[i]=value
    return np.mean(delta)

# Below are the code to draw price&delta against moneyness
def Draw_Prob1(size, S, K, r, sigma, T, X, Rebate):
    x = np.linspace(S - 10, S + 5, size)
    docwr = [0] * size
    dipwr = [0] * size
    anti_rebate= [0] * size
    mon = [0] * size
    for i in range(len(x)):
        docwr[i] = Doc_with_rebate(x[i], K, r, sigma, T, X, Rebate)
        dipwr[i] = Dip_with_rebate(x[i], K, r, sigma, T, X, Rebate)
        anti_rebate[i] = Anti_put_eur_digital_rebate(x[i], K, r, sigma, T, X)
        mon[i] = np.log(x[i] / K)
    plt.plot(mon, docwr, color='black', label='Doc_with_rebate')
    plt.plot(mon, dipwr, color='orange', label='Dip_with_rebate')
    # plt.plot(mon, anti_rebate, color='red', label='Anti-rebate')
    plt.xlabel("Moneyness")
    plt.ylabel("Price")
    plt.legend()
    plt.title('Price with Moneyness')
    plt.show()


def Draw_Prob2(size,S,K,r,sigma,T,X,Rebate):
    x=np.linspace(S-10, S+5, size)
    docwr=[0]*size
    dipwr=[0]*size
    mon=[0]*size
    for i in range(len(x)):
        docwr[i]=delta_of_doc(x[i],K,r,sigma,T,X,Rebate)
        dipwr[i]=delta_of_dip(x[i],K,r,sigma,T,X,Rebate)
        mon[i]=np.log(x[i]/K)
    
    plt.plot(mon, docwr, color='black', label='delta_of_doc')
    plt.plot(mon, dipwr, color='orange', label='delta_of_dip')
    plt.xlabel("Moneyness")
    plt.ylabel("Delta")
    plt.legend()
    plt.title('Delta with Moneyness')
    plt.show()

def Draw_changing_strike(size, S, K, r, sigma, T, X, Rebate):
    x = np.linspace(K - 20, K + 20, size)
    docwr = [0] * size
    dipwr = [0] * size
    eur_call = [0] * size
    eur_put= [0] * size
    for i in range(len(x)):
        docwr[i] = Doc_with_rebate(S, x[i], r, sigma, T, X, Rebate)
        dipwr[i] = Dip_with_rebate(S, x[i], r, sigma, T, X, Rebate)
        eur_call[i] = Call(S, x[i], r, sigma, T)
        eur_put[i] = Put(S, x[i], r, sigma, T)
    plt.plot(x, docwr, color='black', label='Doc_with_rebate')
    plt.plot(x, dipwr, color='orange', label='Dip_with_rebate')
    plt.plot(x, eur_call, color='red', label='Vanilla call')
    plt.plot(x, eur_put, color='blue', label='Vanilla put')
    plt.xlabel("Strike")
    plt.ylabel("Price")
    plt.legend()
    plt.title('Price with changing strike')
    plt.show()

def Draw_changing_barrier(size, S, K, r, sigma, T, X, Rebate):
    x = np.linspace(X - 20, X + 20, size)
    docwr = [0] * size
    dipwr = [0] * size
    eur_call = [0] * size
    eur_put= [0] * size
    for i in range(len(x)):
        docwr[i] = Doc_with_rebate(S, K, r, sigma, T, x[i], Rebate)
        dipwr[i] = Dip_with_rebate(S, K, r, sigma, T, x[i], Rebate)
        eur_call[i] = Call(S, K, r, sigma, T)
        eur_put[i] = Put(S, K, r, sigma, T)
    plt.plot(x, docwr, color='black', label='Doc_with_rebate')
    plt.plot(x, dipwr, color='orange', label='Dip_with_rebate')
    plt.plot(x, eur_call, color='red', label='Vanilla call')
    plt.plot(x, eur_put, color='blue', label='Vanilla put')
    plt.xlabel("Barrier")
    plt.ylabel("Price")
    plt.legend()
    plt.title('Price with changing barrier')
    plt.show()


def main():
    S = 50
    K = 50
    r = 0.02
    sigma = 0.05
    T = 1
    X = 45
    Rebate = 3
    size = 100
    # Question 1 and 2 Price and delta with moneyness
    Draw_Prob1(size, S, K, r, sigma, T, X, Rebate)
    Draw_Prob2(size, S, K, r, sigma, T, X, Rebate)
    # Question 3 Extreme modeling test
    print("Option price for down-out-call with rebate in extreme case: " + str(Doc_with_rebate(S, K, r, sigma, T, S - 10, Rebate)))
    print("Option price of vanilla call: " + str(Call(S, K, r, sigma, T)))
    print("Option price for down-in-put with rebate in extreme case : " + str(Dip_with_rebate(S, K, r, sigma, T, S - 0.001, Rebate)))
    print("Option price by vanilla put: " + str(Put(S, K, r, sigma, T)))
    # Question 4 Alternative method
    print("Option price by Monte Carl for down-out-call: " + str(Monte_doc(X, K, S, T, sigma, r, Rebate)))
    print("Option price by deducted formula: " + str(Doc_with_rebate(S, K, r, sigma, T, X, Rebate)))
    print("Option price by Monte Carl for down-in-put: " + str(Monte_dip(X, K, S, T, sigma, r, Rebate)))
    print("Option price by deducted formula: " + str(Dip_with_rebate(S, K, r, sigma, T, X, Rebate)))
    Draw_changing_strike(size, S, K, r, sigma, T, X, Rebate)
    Draw_changing_strike(size, S, K, r, sigma, T, 49, Rebate)
    Draw_changing_barrier(size, S, K, r, sigma, T, X, Rebate)

    # Code for varying sigma to test Price formula for put_eur_rebate by MC
    for i in range(5, 51, 10):
        i=i/100
        print("The sigma now is " + str(i) )
        print("Option price by Monte Carl for Put-Eur_Rebate is : " + str(Monte_rebate(S, K, r, i, T, X)))
        print("Option price by deducted formula: " + str(Put_eur_digital_rebate(S, K, r, i, T, X)))


    # Code for varying time to maturity to test Price formula for put_eur_rebate by MC
    for i in range(25, 250, 50):
        i=i/100
        print("The time to maturity now is " + str(i) )
        print("Option price by Monte Carl for Put-Eur_Rebate is : " + str(Monte_rebate(S, K, r, sigma, i, X)))
        print("Option price by deducted formula: " + str(Put_eur_digital_rebate(S, K, r, sigma, i, X)))


    # Code for varying barrier to test Price formula for put_eur_rebate by MC
    for i in range(X,X+5):
        print("The barrier now is " + str(i) )
        print("Option price by Monte Carl for Put-Eur_Rebate is : " + str(Monte_rebate(S, K, r, sigma, T, i)))
        print("Option price by deducted formula: " + str(Put_eur_digital_rebate(S, K, r, sigma, T, i)))


if __name__ == "__main__":
    main()
