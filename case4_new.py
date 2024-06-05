# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:35:48 2021

@author: de
"""


###  hyperparameters PCADWT RRL

import numpy as np
import pandas as pd
from models.Molina import  StockTradingStrategy as ST1
from models.MolinaFuture import StockTradingStrategy as ST2
from sklearn.model_selection import ParameterGrid
#from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime as dt


from sklearn.decomposition import PCA
import pywt
import scipy.io
import talib as ta

import time 

start = time.process_time()


######################################## time series for normal feature, n_lagged_time_steps
def constructTimeSeriesWithMultiFeatures(market_info, id_t_end, ):

    df = market_info[:id_t_end]
    
    returnIndex = market_info['Close'].diff()
    # momentum indicators
    mom = ta.MOM(market_info['Close'], timeperiod=14)
    
    macd, macdsignal, macdhist = ta.MACD(market_info['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    mfi = ta.MFI(market_info['High'], market_info['Low'], market_info['Close'], market_info['Volume'], timeperiod=14)
    
    rsi = ta.RSI(market_info['Close'], timeperiod=14)
    
    fastk, fastd = ta.STOCHF(market_info['High'], market_info['Low'], market_info['Close'], fastk_period=14, fastd_period=7, fastd_matype=0)
    
    # volatility indicators
    atr = ta.ATR(market_info['High'], market_info['Low'], market_info['Close'], timeperiod=14)
    natr = ta.NATR(market_info['High'], market_info['Low'], market_info['Close'], timeperiod=14)
    
#    tr = ta.TRANGE(market_info['High'], market_info['Low'], market_info['Close'])
    
    # cycle indicators
    ht_dcperiod = ta.HT_DCPERIOD(market_info['Close'])
    
#    ht_dcphase = ta.HT_DCPHASE(market_info['Close'])
    
#    inphase, quadrature = ta.HT_PHASOR(market_info['Close'])
    
    sine, leadsine = ta.HT_SINE(market_info['Close'])
    
    integer = ta.HT_TRENDMODE(market_info['Close'])
    
    # Volume indicators
    ad = ta.AD(market_info['High'], market_info['Low'], market_info['Close'], market_info['Volume'])
    
    adosc = ta.ADOSC(market_info['High'], market_info['Low'], market_info['Close'], market_info['Volume'], fastperiod=3, slowperiod=10)
    
    obv = ta.OBV(market_info['Close'], market_info['Volume'])
    
    #concatenate technical indicators
    techid = pd.concat([mom, macd, mfi, rsi, atr, natr,  ht_dcperiod, sine, integer, ad, obv], axis=1)
    
    nanid = techid.isna().any(axis=1).tolist().count(True)
    
    techid_trun = techid.iloc[nanid:id_t_end,:]
    
    r_ts = returnIndex[nanid:id_t_end]
    
    p_ts = np.array(df['Close'])[nanid:]     
    
    return  techid_trun.values, r_ts.values, nanid, p_ts

def discrete_wavelet_transform(features, wave_obj, dec_level):
#    wavelet_obj = pywt.Wavelet(wave_obj)
    
    for j in range(features.shape[1]):
        coeffs = pywt.wavedec(features[:,j], wave_obj, mode='periodization', level=dec_level)
        
        for i in range(1, len(coeffs)):
            thd = 2*np.std(coeffs[i])
            dc_denoise = pywt.threshold(coeffs[i], thd, mode='soft', substitute=0)
            coeffs[i] = dc_denoise
            
        feature_denoise = pywt.waverec(coeffs, wave_obj, mode='periodization')
        
        if len(features) % 2 == 0:
            features[:,j] = feature_denoise
        else:
            features[:,j] = feature_denoise[:-1]
            
    return features

def tradingPcadwt1(market_info, 
                 learning_rate, 
                 lambda_re,
                 delta,
                 random_state,
                 n_samples,
                 window_train,
                 dec_level,
                 evr,
                 n_epochs,
                 N):

    X_ts, r_ts, nanid, p_ts = constructTimeSeriesWithMultiFeatures(market_info, n_samples,  )
    
    X_ts = (X_ts - np.mean(X_ts,0))/ np.std(X_ts,0)  # z_score of returns     
    
    pca = PCA(n_components=evr)
    
    X_pca = pca.fit_transform(X_ts)
    tmp = X_pca.copy() ## 
    X_pcadwt = discrete_wavelet_transform(tmp, 'haar', dec_level)
    
#StockTradingStrategy
    test = ST1(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)    

#    listRt_pcadwt, listFt_pcadwt, listSr_pcadwt = test.walkForward(X_pcadwt, r_ts, p_ts, window_train, n_epochs, N) 

    listRt_pcadwt, listFt_pcadwt, listSr_pcadwt = test.walkForward(X_pcadwt, r_ts, window_train, n_epochs, N)                    

    return  listRt_pcadwt, listFt_pcadwt,  nanid  

def tradingPcadwt2(market_info, 
                 learning_rate, 
                 lambda_re,
                 delta,
                 random_state,
                 n_samples,
                 window_train,
                 dec_level,
                 evr,
                 n_epochs,
                 N):

    X_ts, r_ts, nanid, p_ts = constructTimeSeriesWithMultiFeatures(market_info, n_samples,  )
    
    X_ts = (X_ts - np.mean(X_ts,0))/ np.std(X_ts,0)  # z_score of returns     
    
    pca = PCA(n_components=evr)
    
    X_pca = pca.fit_transform(X_ts)
    tmp = X_pca.copy() ## 
    X_pcadwt = discrete_wavelet_transform(tmp, 'haar', dec_level)
    
#StockTradingStrategy
    test = ST2(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)    

    listRt_pcadwt, listFt_pcadwt, listSr_pcadwt = test.walkForward(X_pcadwt, r_ts, p_ts, window_train, n_epochs, N) 

#    listRt_pcadwt, listFt_pcadwt, listSr_pcadwt = test.walkForward(X_pcadwt, r_ts, window_train, n_epochs, N)                    

    return  listRt_pcadwt, listFt_pcadwt,  nanid  

#fname = 'data_old/szzs.csv'
#fname = 'data_old/nyse.csv'
 
fnames = ['data_old/nyse.csv', 'data_old/xom.csv', 'data_old/zcf.csv']

sr_all = []

for fname in fnames:
    
    if fname == 'data_old/nyse.csv':
        tc = 1
    elif fname == 'data_old/xom.csv':
        tc = 0.01
    else:
        tc = 0.001   
        
#    rs = [np.random.randint(100000, size=1).item() for _ in range(20)]    
    param_pcadwt = { 'random_state': [42], 'learning_rate': [0.1], 'lambda_re':  [0.01], 
    'delta': [tc] , 'n_samples': [6000], 'window_train': [500], 'dec_level': [1,2,3,4,5,6,7,8], 'evr':[0.95],
    'n_epochs': [100], 'N':  [500]}    
    
    df = pd.read_csv(fname, header=None)

    time_str = df[0] #+" " + df[1]
#dates = [dt.strptime(time_str[i], '%Y/%m/%d %H:%M') for i in range(len(time_str))]
#dates = [dt.strptime(time_str[i], '%Y/%m/%d') for i in range(len(time_str))]
    dates = [dt.strptime(time_str[i], '%Y-%m-%d') for i in range(len(time_str))]
    market_info = df.iloc[:,1:]
    market_info.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    market_info.index = dates

    list_sr = []
    if fname != 'data_old/zcf.csv':
        for it in ParameterGrid(param_pcadwt):
        
            listRt_pcadwt, listFt_pcadwt, nanid  = tradingPcadwt1(market_info, **it)  
    
            np_pcadwt = np.sum(listRt_pcadwt)

            sr_pcadwt = np.mean(listRt_pcadwt) / np.std(listRt_pcadwt) * np.sqrt(252) 
            list_sr.append(sr_pcadwt)
    
            print('np:', round(np_pcadwt,2),'\n', 'sr:', round(sr_pcadwt,2), '\n', )
    else:
        for it in ParameterGrid(param_pcadwt):
            listRt_pcadwt, listFt_pcadwt, nanid  = tradingPcadwt2(market_info, **it)  
    
            np_pcadwt = np.sum(listRt_pcadwt)

            sr_pcadwt = np.mean(listRt_pcadwt) / np.std(listRt_pcadwt) * np.sqrt(252) 
            list_sr.append(sr_pcadwt)
    
            print('np:', round(np_pcadwt,2),'\n', 'sr:', round(sr_pcadwt,2), '\n', )
           
    sr_all.append(list_sr)
    
end = time.process_time()
print('running time:%s'%(end-start))

def plot_hypers(x, sr_all):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    
    ax.plot(x, sr_all[0], '-bo', label='NYSE')
    ax.plot(x, sr_all[1], '-go', label='XOM')
    ax.plot(x, sr_all[2], '-ko', label='ZCF')
    
    ax.legend(loc = 'upper left')
 #   ax.set_xlabel('Decomposition Levels')
#    ax.set_xlabel('Explained Variance Ratio')
    ax.set_xlabel('Trading Periods')
#    ax.set_xlabel('Training Epochs')
#    ax.set_xlabel('Training Windows')
#    ax.set_xlabel('Transaction Costs')
    ax.set_ylabel('Sharpe Ratio')
    
#x = param_pcadwt['dec_level']
x = param_pcadwt['N']    
plot_hypers(x, sr_all)

'''
def plot_box(sr_all):
      
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
     # den_cps = np.sum(den_all, axis=1)
     # red_cps = np.sum(red_all, axis=1)
     # tec_cps = np.sum(tec_all, axis=1)
     # srs = [sr_all[0], sr_all[1], srs[2]]
    ax.boxplot(sr_all, labels=['NYSE', 'XOM', 'ZCF'], notch=True, sym='rx', vert=True)
    ax.set_xlabel('Data Sets')
    ax.set_ylabel('Sharpe Ratio')
plot_box(sr_all)
'''

'''
evr = np.load('D:/A_Mannheim/Quanttrade/results_new/4evr.npy')
dl = np.load('D:/A_Mannheim/Quanttrade/results_new/4dl.npy')
iniw = np.load('D:/A_Mannheim/Quanttrade/results_new/4iniw.npy').tolist()
lr = np.load('D:/A_Mannheim/Quanttrade/results_new/4lr.npy')
epo = np.load('D:/A_Mannheim/Quanttrade/results_new/4epo.npy')
trwin = np.load('D:/A_Mannheim/Quanttrade/results_new/4trwin.npy')
tdwin = np.load('D:/A_Mannheim/Quanttrade/results_new/4tdwin.npy')
tc = np.load('D:/A_Mannheim/Quanttrade/results_new/4tc.npy')

def plot_all(evr, dl, iniw, lr, epo, trwin, tdwin, tc):
    
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
    
    x = [0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]    
    ax[0,0].plot(x, evr[0], '-bo', label='NYSE')
    ax[0,0].plot(x, evr[1], '-go', label='XOM')
    ax[0,0].plot(x, evr[2], '-ko', label='ZCF')
    ax[0,0].legend(loc = 'upper left')
    ax[0,0].set_xlabel('Explained Variance Ratio')
    ax[0,0].set_ylabel('Annualized Sharpe Ratio')

    x = [1,2,3,4,5,6,7,8]    
    ax[0,1].plot(x, dl[0], '-bo', label='NYSE')
    ax[0,1].plot(x, dl[1], '-go', label='XOM')
    ax[0,1].plot(x, dl[2], '-ko', label='ZCF')
    ax[0,1].legend(loc = 'upper left')
    ax[0,1].set_xlabel('Decomposition Level')
    ax[0,1].set_ylabel('Annualized Sharpe Ratio')

    ax[0,2].boxplot(iniw, labels=['NYSE', 'XOM', 'ZCF'], notch=True, sym='rx', vert=True)
    ax[0,2].set_xlabel('Data Set')
    ax[0,2].set_ylabel('Annualized Sharpe Ratio')
    
    x = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]    
    ax[0,3].plot(x, lr[0], '-bo', label='NYSE')
    ax[0,3].plot(x, lr[1], '-go', label='XOM')
    ax[0,3].plot(x, lr[2], '-ko', label='ZCF')
    ax[0,3].legend(loc = 'upper left')
    ax[0,3].set_xlabel('Learning Rate')
    ax[0,3].set_ylabel('Annualized Sharpe Ratio')  
    
    x = [50,100,150,200,250,300,350,400]    
    ax[1,0].plot(x, epo[0], '-bo', label='NYSE')
    ax[1,0].plot(x, epo[1], '-go', label='XOM')
    ax[1,0].plot(x, epo[2], '-ko', label='ZCF')
    ax[1,0].legend(loc = 'upper left')
    ax[1,0].set_xlabel('Epoch')
    ax[1,0].set_ylabel('Annualized Sharpe Ratio')   
    
    x = [100,200,300,400,500,600,700,800]    
    ax[1,1].plot(x, trwin[0], '-bo', label='NYSE')
    ax[1,1].plot(x, trwin[1], '-go', label='XOM')
    ax[1,1].plot(x, trwin[2], '-ko', label='ZCF')
    ax[1,1].legend(loc = 'upper left')
    ax[1,1].set_xlabel('Training Period')
    ax[1,1].set_ylabel('Annualized Sharpe Ratio')     
    
    x = [100,200,300,400,500,600,700,800]    
    ax[1,2].plot(x, tdwin[0], '-bo', label='NYSE')
    ax[1,2].plot(x, tdwin[1], '-go', label='XOM')
    ax[1,2].plot(x, tdwin[2], '-ko', label='ZCF')
    ax[1,2].legend(loc = 'upper left')
    ax[1,2].set_xlabel('Trading Period')
    ax[1,2].set_ylabel('Annualized Sharpe Ratio')  
    
    x = [0,0.005,0.010,0.015,0.020,0.025,0.030,0.035]    
    ax[1,3].plot(x, tc[0], '-bo', label='NYSE')
    ax[1,3].plot(x, tc[1], '-go', label='XOM')
    ax[1,3].plot(x, tc[2], '-ko', label='ZCF')
    ax[1,3].legend(loc = 'upper left')
    ax[1,3].set_xlabel('Transaction Cost')
    ax[1,3].set_ylabel('Annualized Sharpe Ratio')   
    
    plt.savefig('D:/A_Mannheim/Quanttrade/results_new/4all.eps', dpi=600, bbox_inches='tight')
plot_all(evr, dl, iniw, lr, epo, trwin, tdwin, tc)  
'''
  