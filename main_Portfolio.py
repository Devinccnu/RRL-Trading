# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:41:47 2019

@author: lilb
"""

import numpy as np
import pandas as pd
from models.Portfolio_Selection import  Portfolio_Selection_Strategy
#from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime as dt
import time
#start_time = time.time()
#def read_return_series(fname):
     
     
def construct_features(mar_info, lag):
    df = mar_info[:, 1:]/mar_info[:, :-1] -1
#    df1 = forex1[1:]/forex1[:-1]-1 #np.diff(forex1) #
#    df2 = forex2[1:]/forex2[:-1]-1 #np.diff(forex2) #
#    df3 = forex3[1:]/forex3[:-1]-1 #np.diff(forex3) #
    m = df.shape[1]
#    m = len(df1)
#    df = np.vstack((df1, df2, df3))
    
    X_ts = np.zeros((m-lag, 3, lag))
    r_ts = np.zeros((3, m))
    for i in range(m-lag) :
        for j in range(3) :
            X_ts[i,j,:] = df[j, i:(i+lag)]
            
    r_ts = df[:, lag-1:-1]
#    print(X_ts.shape)
#    print(r_ts.shape)
    return  X_ts, r_ts

def RRL_Portfolio(mar_info, 
                  window_train=1000, 
                  window_test=1000, 
                  lag=8, 
                  learning_rate=0.001, 
                  lambda_re=0.001, 
                  delta=0.005, 
                  n_epochs=100,
                  random_state=42):
    
    X_ts, r_ts = construct_features(mar_info, lag)
    
    for j in range(3):
        
        scaler = preprocessing.StandardScaler().fit(X_ts[:,j,:window_train])   #for RRL
        X_ts[:,j,:] = scaler.transform(X_ts[:,j,:]) 
    
    test = Portfolio_Selection_Strategy(learning_rate=learning_rate,
                                        lambda_re=lambda_re,
                                        delta=delta,  
                                        random_state=random_state)  
    
#    list_SR
    start_time = time.time()
    test_o  = test.fit(X_ts[:window_train,:,:], r_ts[:,:window_train], n_epochs)
     
    list_Rt, list_Ft, SR = test_o.evaluate(X_ts[window_train:window_train + window_test,:,:], 
                                         r_ts[:,window_train:window_train + window_test])
    print('%s seconds'%(time.time()-start_time)) 
    
#    test_logsig = test.fit_logsig(X_ts[:window_train,:,:], r_ts[:,:window_train], n_epochs)
    
#    list_Rt_log, list_Ft_log, SR_log = test_logsig.evaluate_logsig(X_ts[window_train:window_train + window_test,:,:], 
#                                         r_ts[:,window_train:window_train + window_test])    
#    start_time = time.time()
#    
#    test_mini1 = test.mini1_fit(X_ts[:window_train,:,:], r_ts[:,:window_train], n_epochs)
#
#    list_Rt_mini1, list_Ft_mini1, SR_mini1 = test_mini1.evaluate(X_ts[window_train:window_train + window_test,:,:], 
#                                         r_ts[:,window_train:window_train + window_test])
#    print('%s seconds'%(time.time()-start_time))
#    start_time = time.time()
#    test_mini2 = test.mini2_fit(X_ts[:window_train,:,:], r_ts[:,:window_train], n_epochs)
#
#    list_Rt_mini2, list_Ft_mini2, SR_mini2 = test_mini2.evaluate(X_ts[window_train:window_train + window_test,:,:], 
#                                         r_ts[:,window_train:window_train + window_test])  
#    print('%s seconds'%(time.time()-start_time))
#    start_time = time.time()
#    test_mini3 = test.mini3_fit(X_ts[:window_train,:,:], r_ts[:,:window_train], n_epochs)
#
#    list_Rt_mini3, list_Ft_mini3, SR_mini3 = test_mini3.evaluate(X_ts[window_train:window_train + window_test,:,:], 
#                                         r_ts[:,window_train:window_train + window_test])   
#    print('%s seconds'%(time.time()-start_time))
    

    return  r_ts, list_Rt, list_Ft, SR, #list_Rt_log, list_Ft_log, SR_log #list_SR # list_SR, list_Rt_mini1, SR_mini1, list_Rt_mini2, SR_mini2, list_Rt_mini3, SR_mini3, 
#    list_Rt_mini,
def plot_figure(mar_info, bah, marko_sr, list_Rt,list_Ft, window_train, window_test):

     Rs = np.asarray(list_Rt)
     Fs = np.asarray(list_Ft)

     fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(9, 15))
     t = np.arange(window_train,window_train+window_test,1)  #9991  truncate series len
     
     ax[0].plot(t, mar_info[0][window_train:window_train+window_test])
     ax[0].plot(t, mar_info[1][window_train:window_train+window_test])
     ax[0].plot(t, mar_info[2][window_train:window_train+window_test])
     ax[0].set_xlabel('Time')
     ax[0].set_ylabel('Price')  #['Close']
   
     ax[1].plot(t, np.cumprod(Rs[:len(t)])-1, label='DPS-RRL')
#     ax[1].plot(t, np.cumprod(Rs_mini[:len(t)])-1, label='rrl_mini')
     ax[1].plot(t, np.cumprod(bah[window_train:window_train+window_test])-1, label='EW-B&H')
     ax[1].plot(t, np.cumprod(marko_sr[window_train:window_train+window_test])-1, label='Max-SR-B&H')
     ax[1].legend(loc="upper left")
     ax[1].set_xlabel('Time')
     ax[1].set_ylabel('Cumulative Profits')

     ax[2].plot(t, Fs[:len(t),0])
     ax[2].set_xlabel('Time')
     ax[2].set_ylabel('Trading Signal of Security 1')
     ax[2].set_ylim(0,1.0) 
         
     ax[3].plot(t, Fs[:len(t),1])
     ax[3].set_xlabel('Time')
     ax[3].set_ylabel('Trading Signal of Security 2')
     ax[3].set_ylim(0,1.0)  
     
     ax[4].plot(t, Fs[:len(t),2])
     ax[4].set_xlabel('Time')
     ax[4].set_ylabel('Trading Signal of Security 3')
     ax[4].set_ylim(0,1.0)     
     
def plot_price_arti(mar_info, window_train, window_test):
#    window_train
     t = np.arange(0, window_train+window_test, 1)  #9991  truncate series len
     plt.figure(figsize=(10,5))      
     plt.plot(t, mar_info[0,:window_train+window_test], '-', label='Security 1')              
     plt.plot(t, mar_info[1,:window_train+window_test], '--', label='Security 2')  
     plt.plot(t, mar_info[2,:window_train+window_test], ':', label='Security 3')  
     
     plt.legend(loc="upper left")
     plt.xlabel('Time', fontsize =13)
     plt.ylabel('Price', fontsize=13)

     plt.savefig('arti_price.pdf', format='pdf', dpi=300, bbox_inches='tight')
     
def plot_result_arti(mar_info, bah, marko_sr, list_Rt, window_train, window_test):
     Rs = np.asarray(list_Rt)

     fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
     t = np.arange(window_train,window_train+window_test,1)  #9991  truncate series len
     
     ax[0].plot(t, mar_info[0][window_train:window_train+window_test], label='Security 1')
     ax[0].plot(t, mar_info[1][window_train:window_train+window_test], label='Security 2')
     ax[0].plot(t, mar_info[2][window_train:window_train+window_test], label='Security 3')
     ax[0].set_xlabel('Time')
     ax[0].set_ylabel('Price')
     ax[0].legend(loc="upper left")
     
     ax[1].plot(t, np.cumprod(Rs[:len(t)])-1, label='DPS-RRL')
     ax[1].plot(t, np.cumprod(bah[window_train:window_train+window_test])-1, label='EW-B&H')
     ax[1].plot(t, np.cumprod(marko_sr[window_train:window_train+window_test])-1, label='Max-SR-B&H')
     ax[1].legend(loc="upper left")
     ax[1].set_xlabel('Time')
     ax[1].set_ylabel('Cumulative Profits')
     
     plt.savefig('arti_result.eps', format='eps', dpi=1200, bbox_inches='tight')

def plot_Ft_arti(list_Ft, window_train, window_test):
     Fs = np.asarray(list_Ft)
     
     fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
     t = np.arange(window_train,window_train+window_test,1)     
#window_train
     ax[0].plot(t, Fs[:len(t),0])
#     ax[0].set_xlabel('Time', fontsize=13)
#     ax[0].set_ylabel('Trading Signal of Security 1', fontsize=13)
     ax[0].set_ylim(0,1.0) 
#         window_train
     ax[1].plot(t, Fs[:len(t),1])
#     ax[1].set_xlabel('Time', fontsize=13)
#     ax[1].set_ylabel('Trading Signal of Security 2', fontsize=13)
     ax[1].set_ylim(0,1.0)  
#     window_train
     ax[2].plot(t, Fs[:len(t),2])
#     ax[2].set_xlabel('Time', fontsize=13)
#     ax[2].set_ylabel('Trading Signal of Security 3', fontsize=13)
#     ax[2].set_ylim(0,1.0)   
     fig.text(0.5,0.04,'Time', ha='center', va='center', fontsize=13)
     fig.text(0.06,0.5, 'Trading Signals of Securities', ha='center', va='center', rotation='vertical', fontsize=13)

     plt.savefig('arti_signal.pdf', format='pdf', dpi=300, bbox_inches='tight')
     
def plot_mini(bah, marko_sr, list_Rt, list_Rt_mini1,list_Rt_mini2,list_Rt_mini3, window_train, window_test):
     Rs_mini1 = np.asarray(list_Rt_mini1)
     Rs_mini2 = np.asarray(list_Rt_mini2)
     Rs_mini3 = np.asarray(list_Rt_mini3)     
     Rs = np.asarray(list_Rt)
#     Rs2 = np.asarray(list_Rt2)

     
     t = np.arange(window_train,window_train+window_test,1)  #9991  truncate series len
     plt.figure(figsize=(10,5))      
     plt.plot(t, np.cumprod(Rs[:len(t)])-1, label='DPS-RRL')
     plt.plot(t, np.cumprod(Rs_mini1[:len(t)])-1, label='BS1-OPS-RRL')
     plt.plot(t, np.cumprod(Rs_mini2[:len(t)])-1, label='BS2-OPS-RRL')
     plt.plot(t, np.cumprod(Rs_mini3[:len(t)])-1, label='BS3-OPS-RRL')   
     
     plt.plot(t, np.cumprod(bah[window_train:window_train+window_test])-1, label='EW-B&H')
     plt.plot(t, np.cumprod(marko_sr[window_train:window_train+window_test])-1, label='Max-SR-B&H')

     plt.legend(loc="upper left")
     plt.xlabel('Time')
     plt.ylabel('Cumulative Profits')

     plt.savefig('mini_bs_opsrrl.eps', format='eps', dpi=1200, bbox_inches='tight')
#     bah, marko_sr,
def plot_logsig(list_Rt, list_Rt_log, window_train, window_test):
     Rs_log = np.asarray(list_Rt_log)  
     Rs = np.asarray(list_Rt)
#     Rs2 = np.asarray(list_Rt2)
     
     t = np.arange(window_train,window_train+window_test,1)  #9991  truncate series len
     plt.figure(figsize=(10,7))      
     plt.plot(t, np.cumprod(Rs[:len(t)])-1, '-', label='Tanh-DPS-RRL')
     plt.plot(t, np.cumprod(Rs_log[:len(t)])-1, '--', label='Logistic-DPS-RRL') 
     
#     plt.plot(t, np.cumprod(bah[window_train:window_train+window_test])-1, label='EW-B&H')
#     plt.plot(t, np.cumprod(marko_sr[window_train:window_train+window_test])-1, label='Max-SR-B&H')

     plt.legend(loc="upper left")
     plt.xlabel('Time', fontsize=13)
     plt.ylabel('Cumulative Profits', fontsize=13)
     plt.savefig('result_logsig.pdf', format='pdf', dpi=300, bbox_inches='tight')
     
def plot_logsig_Ft(list_Ft, list_Ft_log, window_train, window_test):
     Fs = np.asarray(list_Ft)
     Fs_log = np.asarray(list_Ft_log)
     
     t = np.arange(window_train,window_train+window_test,1)  #9991  truncate series len
     fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
     ax[0].plot(t, Fs[:len(t),0], label='DPS-RRL')
     ax[0].plot(t, Fs_log[:len(t),0], label='Log-OPS-RRL')
     ax[0].set_xlabel('Time')
     ax[0].set_ylabel('Trading Signal of Security 1')
     ax[0].set_ylim(0,1.0) 
     ax[0].legend(loc="upper left")
         
     ax[1].plot(t, Fs[:len(t),1], label='DPS-RRL')
     ax[1].plot(t, Fs_log[:len(t),1], label='Log-OPS-RRL')     
     ax[1].set_xlabel('Time')
     ax[1].set_ylabel('Trading Signal of Security 2')
     ax[1].set_ylim(0,1.0)  
     ax[1].legend(loc="upper left")
     
     ax[2].plot(t, Fs[:len(t),2], label='DPS-RRL')
     ax[2].plot(t, Fs_log[:len(t),2], label='Log-OPS-RRL')     
     ax[2].set_xlabel('Time')
     ax[2].set_ylabel('Trading Signal of Security 3')
     ax[2].set_ylim(0,1.0)     
     ax[2].legend(loc="upper left")     
     
def plot_price_real(mar_info, window_train, window_test):
#    window_train
     t = np.arange(0, window_train+window_test, 1)  #9991  truncate series len
     plt.figure(figsize=(10,5))      
     plt.plot(t, mar_info[0,:window_train+window_test], '-', label='IWC')              
     plt.plot(t, mar_info[1,:window_train+window_test], '--', label='IWD')  
     plt.plot(t, mar_info[2,:window_train+window_test], ':', label='SPY')  
     
     plt.legend(loc="upper left")
     plt.xlabel('Time', fontsize=13)
     plt.ylabel('Price', fontsize=13)

     plt.savefig('real_price.pdf', format='pdf', dpi=300, bbox_inches='tight')
     
def plot_result_real(bah, marko_sr, list_Rt, window_train, window_test):
     
     Rs = np.asarray(list_Rt)
#    0
     t = np.arange(window_train,window_train+window_test,1)  #9991  truncate series len
     plt.figure(figsize=(10,5))      
     plt.plot(t, np.cumprod(Rs[:len(t)])-1, label='OPS-RRL')     
     plt.plot(t, np.cumprod(bah[window_train:window_train+window_test])-1, label='EW-B&H')
     plt.plot(t, np.cumprod(marko_sr[window_train:window_train+window_test])-1, label='Max-SR-B&H')

     plt.legend(loc="upper left")
     plt.xlabel('Time')
     plt.ylabel('Cumulative Profits')

     plt.savefig('real_result.eps', format='eps', dpi=1200, bbox_inches='tight')
     
def plot_Ft_real(list_Ft, window_train, window_test):
     Fs = np.asarray(list_Ft)
     
     fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
     t = np.arange(window_train,window_train+window_test,1)     
#window_train
     ax[0].plot(t, Fs[:window_train+window_test,0])
#     ax[0].set_xlabel('Time',fontsize=13)
#     ax[0].set_ylabel('Trading Signal of IWC', fontsize=13)
     ax[0].set_ylim(0,1.0) 
#         window_train
     ax[1].plot(t, Fs[:window_train+window_test,1])
#     ax[1].set_xlabel('Time', fontsize=13)
#     ax[1].set_ylabel('Trading Signal of IWD', fontsize=13)
     ax[1].set_ylim(0,1.0)  
#     window_train
     ax[2].plot(t, Fs[:window_train+window_test,2])
#     ax[2].set_xlabel('Time', fontsize=13)
#     ax[2].set_ylabel('Trading Signal of SPY', fontsize=13)
     ax[2].set_ylim(0,1.0)   
     fig.text(0.5,0.04,'Time', ha='center', va='center', fontsize=13)
     fig.text(0.06,0.5, 'Trading Signals of Securities', ha='center', va='center', rotation='vertical')
     plt.savefig('real_signal.pdf', format='pdf', dpi=300, bbox_inches='tight')
     
random_state = 42
#print(random_state)   

fname = ['data/z1.csv', 'data/z2.csv', 'data/z3.csv']
for i in range(len(fname)):
     df = pd.read_csv(fname[i], header=None)
     if i == 0:
         market_info = np.zeros((len(fname), len(df)))     
     market_info[i,:] = np.array(df[0])
     
###random_state = np.asscalar(np.random.randint(100000, size=1)) # generate random seed
## 
window_train = 1000  #1000
window_test =  2000 - window_train - 10 #1000

     
#list_cp = []
#list_sr = []
#with open('results/port_cp.5%.csv', 'w') as f:
#    with open('results/port_sr.5%.csv', 'w') as g:    
#        
#        for i in range(100):
#            
#            random_state = np.asscalar(np.random.randint(100000, size=1))
#            print(random_state)
###   
###list_SR, list_Rt_mini1, SR_mini1, list_Rt_mini2, SR_mini2, list_Rt_mini3, SR_mini3, list_Rt_log, list_Ft_log, SR_log,
r_ts, list_Rt,list_Ft, SR, = RRL_Portfolio(market_info, 
                        window_train = window_train, 
                        window_test = window_test, 
                        lag = 12, 
                        learning_rate = 0.3, 
                        lambda_re = 0.01, 
                        delta = 0.005, 
                        n_epochs = 100, 
                        random_state = random_state )
#plot_logsig(list_Rt, list_Rt_log, window_train, window_test)
#            cp = np.prod(list_Rt)-1
##print('Testing Cumulative Profit \t Sharpe ratio : {:.4f}'.format(np.prod(list_Rt)-1, SR))
#            
#            list_cp.append(cp)
#            list_sr.append(SR)
            
##            print('No.{} \t Testing CP :{:.4f} \t Testing SR :{:.4f}'.format(i, cp, SR))
#
#
#        np.asarray(list_sr)
#        np.savetxt(g, list_sr,  fmt='%1.4f')
##        np.savetxt(g, newline='\n', fmt='%s') 
#            
#    np.asarray(list_cp)
#    np.savetxt(f, list_cp,  fmt='%1.4f')
##    np.savetxt(f, newline='\n', fmt='%s') 
        
         
bah = 1/3 * np.sum(r_ts, axis=0) +1
a = np.mean(bah[window_train:window_train+window_test]-1)
b = np.mean((bah[window_train:window_train+window_test]-1)**2)
sr_bah = a/np.sqrt(b-a**2)
cp_bah = np.prod(bah[window_train:window_train+window_test])-1

max_sr = np.vstack((0.00102*r_ts[0,:], 0.80606*r_ts[1,:], 0.19292*r_ts[2,:]))  # 0.493497, 0.0228992, 0.483603
smax_sr = np.sum(max_sr, axis=0) +1
c = np.mean(smax_sr[window_train:window_train+window_test] -1)
d = np.mean((smax_sr[window_train:window_train+window_test]-1)**2)
sr_max_sr = c/np.sqrt(d-c**2)
cp_max_sr = np.prod(smax_sr[window_train:window_train+window_test])-1

max_ret = np.vstack((0.00961*r_ts[0,:], 0.95402*r_ts[1,:], 0.03637*r_ts[2,:]))
smax_ret = np.sum(max_ret, axis=0) + 1
e = np.mean(smax_ret[window_train:window_train+window_test] -1)
f = np.mean((smax_ret[window_train:window_train+window_test]-1)**2)
sr_max_ret = e/np.sqrt(f-e**2)
cp_max_ret = np.prod(smax_ret[window_train:window_train+window_test])-1

min_var = np.vstack((0.23088*r_ts[0,:], 0.26743*r_ts[1,:], 0.50169*r_ts[2,:]))
smin_var = np.sum(min_var, axis=0) + 1
g = np.mean(smin_var[window_train:window_train+window_test] -1)
h = np.mean((smin_var[window_train:window_train+window_test]-1)**2)
sr_min_var = g/np.sqrt(h-g**2)
cp_min_var = np.prod(smin_var[window_train:window_train+window_test])-1

print('Test Set Sharpe Ratio : {:.4f}'.format(SR))
print('bah Sharpe Ratio : {:.4f}'.format(sr_bah))
print('max_sr Sharpe Ratio : {:.4f}'.format(sr_max_sr))
print('max_ret Sharpe Ratio : {:.4f}'.format(sr_max_ret))
print('min_var Sharpe Ratio : {:.4f}'.format(sr_min_var))

print('Test Cumulative Profit : {:.4f}'.format(np.prod(list_Rt)-1))
print('Buy and Hold Profit : {:.4f}'.format(cp_bah))
print('Max_sr profit : {:.4f}'.format(cp_max_sr))
print('Max_ret profit : {:.4f}'.format(cp_max_ret))
print('Min_var profit : {:.4f}'.format(cp_min_var))

###plot_figure(market_info, bah, marko_sr, list_Rt,list_Ft, window_train, window_test)
#plot_price_arti(market_info, window_train, window_test)
#plot_result_arti(market_info, bah, marko_sr, list_Rt, window_train, window_test)
#plot_Ft_arti(list_Ft, window_train, window_test)
    
#fname = ["data/IWC.csv", 'data/IWD.csv', 'data/SPY.csv']
#
#for i in range(len(fname)):
#     df = pd.read_csv(fname[i], header=None)
#     time_str = df[0] #+" " + df[1]
#     dates = [dt.strptime(time_str[j], '%Y/%m/%d') for j in range(len(time_str))]
#     forex_market_info = df.iloc[:,1:]
#     forex_market_info.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
#     forex_market_info.index = dates
#     if i == 0:
#         market_info = np.zeros((len(fname), len(time_str)))
#         
#     market_info[i,:] = np.array(forex_market_info['Close'])
#train_w = 1400 # 3 years weekly return
#test_w = market_info.shape[1] - train_w - 26#24 
#
##  list_SR
#r_ts, list_Rt,list_Ft, SR = RRL_Portfolio(market_info, 
#                        window_train = train_w, #1000, 
#                        window_test = test_w, #1000, 
#                        lag = 20, 
#                        learning_rate = 0.3, 
#                        lambda_re = 0.01, 
#                        delta = 0.00, 
#                        n_epochs = 200, 
#                        random_state = random_state )
#
#bah = 1/3 * np.sum(r_ts, axis=0) +1
#a = np.mean(bah[train_w:train_w+test_w]-1)
#b = np.mean((bah[train_w:train_w+test_w]-1)**2)
#sr_bah = a/np.sqrt(b-a**2)
#cp_bah = np.prod(bah[train_w:train_w + test_w])-1
#
#max_sr = np.vstack((0.97193*r_ts[0,:], 0.00461*r_ts[1,:], 0.02346*r_ts[2,:]))  # 0.493497, 0.0228992, 0.483603
#smax_sr = np.sum(max_sr, axis=0) +1
#c = np.mean(smax_sr[train_w:train_w+test_w] -1)
#d = np.mean((smax_sr[train_w:train_w+test_w]-1)**2)
#sr_max_sr = c/np.sqrt(d-c**2)
#cp_max_sr = np.prod(smax_sr[train_w:train_w+test_w])-1
#
#max_ret = np.vstack((0.98012*r_ts[0,:], 0.01661*r_ts[1,:], 0.00326*r_ts[2,:]))
#smax_ret = np.sum(max_ret, axis=0) + 1
#e = np.mean(smax_ret[train_w:train_w+test_w] -1)
#f = np.mean((smax_ret[train_w:train_w+test_w]-1)**2)
#sr_max_ret = e/np.sqrt(f-e**2)
#cp_max_ret = np.prod(smax_ret[train_w:train_w+test_w])-1
#
#min_var = np.vstack((0.00410*r_ts[0,:], 0.01661*r_ts[1,:], 0.00326*r_ts[2,:]))
#smin_var = np.sum(min_var, axis=0) + 1
#g = np.mean(smin_var[train_w:train_w+test_w] -1)
#h = np.mean((smin_var[train_w:train_w+test_w]-1)**2)
#sr_min_var = g/np.sqrt(h-g**2)
#cp_min_var = np.prod(smin_var[train_w:train_w+test_w])-1
#
#print('Test Set Sharpe Ratio : {:.4f}'.format(SR))
#print('bah Sharpe Ratio : {:.4f}'.format(sr_bah))
#print('max_sr Sharpe Ratio : {:.4f}'.format(sr_max_sr))
#print('max_ret Sharpe Ratio : {:.4f}'.format(sr_max_ret))
#print('min_var Sharpe Ratio : {:.4f}'.format(sr_min_var))
#
#print('Test Cumulative Profit : {:.4f}'.format(np.prod(list_Rt)-1))
#print('Buy and Hold Profit : {:.4f}'.format(cp_bah))
#print('Max_sr profit : {:.4f}'.format(cp_max_sr))
#print('Max_ret profit : {:.4f}'.format(cp_max_ret))
#print('Min_var profit : {:.4f}'.format(cp_min_var))


#plot_price_real(market_info, window_train=train_w, window_test=test_w)
#plot_result_real(bah, marko_sr, list_Rt, window_train=train_w, window_test=test_w)
#plot_Ft_real(list_Ft, window_train=train_w, window_test=test_w) 

##plot_figure(bah, smax_sr,list_Rt,list_Ft, list_Rt_mini1,list_Rt_mini2,list_Rt_mini3,window_train=1000, window_test=1000)  # ['Close']
 
#plot_mini(bah, marko_sr, list_Rt, list_Rt_mini1,list_Rt_mini2,list_Rt_mini3, window_train=1000, window_test=1000)
#plot_logsig(bah, marko_sr, list_Rt,  list_Rt_log,  window_train=1000, window_test=1000)
#plot_logsig_Ft(list_Ft, list_Ft_log, window_train=1000, window_test=1000)
#print('Test Set Sharpe Ratio logsig : {:.4f}'.format(SR_log))
#print('Test Cumulative Profit logsig : {:.4f}'.format(np.prod(list_Rt_log)-1))

#print('Testing Set Sharpe Ratio : {:.4f}'.format(SR))
#print('bah Sharpe Ratio : {:.4f}'.format(sr_bah))
#print('marko_sr Sharpe Ratio : {:.4f}'.format(sr_marko))
#
#print('Test Cumulative Profit : {:.4f}'.format(np.prod(list_Rt)-1))
#print('Buy and Hold Profit : {:.4f}'.format(np.prod(bah[train_w:])-1))
#print('Markowitz sr profit : {:.4f}'.format(np.prod(marko_sr[train_w:])-1))

#print('Test Set Sharpe Ratio mini1 : {:.4f}'.format(SR_mini1))
#print('Test Cumulative Profit mini1 : {:.4f}'.format(np.prod(list_Rt_mini1)-1))
#
#print('Test Set Sharpe Ratio mini2 : {:.4f}'.format(SR_mini2))
#print('Test Cumulative Profit mini2 : {:.4f}'.format(np.prod(list_Rt_mini2)-1))
#
#print('Test Set Sharpe Ratio mini3 : {:.4f}'.format(SR_mini3))
#print('Test Cumulative Profit mini3 : {:.4f}'.format(np.prod(list_Rt_mini3)-1))

#print('%s seconds'%(time.time()-start_time))





#tf = [list_Ft[i-1]==list_Ft[i] for i in np.arange(1,len(list_Ft)) ]
#print(tf.count(0)) 

#plt.plot(list_SR)
#plt.xlabel('Epoch', fontsize=13)
#plt.ylabel('Sharpe ratio', fontsize=13)
#plt.savefig('SR.pdf', format='pdf', dpi=300, bbox_inches='tight')
##plt.grid()
#plt.show()   


