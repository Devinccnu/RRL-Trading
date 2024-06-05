
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:20:12 2021

@author: de
"""

import numpy as np
import pandas as pd
from models.Molina import  StockTradingStrategy
#from models.MolinaFuture import StockTradingStrategy
#from sklearn.model_selection import ParameterGrid
#from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime as dt
#from SignatureFunctions import time_joint_path, sig_cal_paths, lead_lag, generateSig

from sklearn.decomposition import PCA
import pywt
#from scipy.signal import fftconvolve
#import scipy.io
import talib as ta

def lookbackFeatures(market_info, id_t_end, n_lagged_time_steps):
#    df = market_info[:id_t_end]
#    pt_close = df['Close'] 
#    returnIndex = np.diff(pt_close) # r(t) = p(t) - p(t-1)
    returnIndex = market_info['Close'].diff()
    re_ts = returnIndex.values[1:]  ## first entry is nan
     
    X_ts = np.zeros((len(re_ts) - n_lagged_time_steps, n_lagged_time_steps))
    for i in range(X_ts.shape[0]):
        X_ts[i,:] = re_ts[np.arange(i, i + n_lagged_time_steps,1)]
        
    r_ts = re_ts[n_lagged_time_steps-1:-1]
    nanid = 0
    return X_ts, r_ts, nanid # p_ts

######################################## time series for normal feature, n_lagged_time_steps
def constructTimeSeriesWithMultiFeatures(market_info, id_t_end, ):
# =============================================================================
#     df = market_info[:id_t_end]
#     pt_close = df['Close'] 
#     returnIndex = np.diff(pt_close) # r(t) = p(t) - p(t-1)
# 
#     X_ts = np.zeros((len(pt_close) - n_lagged_time_steps-1, n_lagged_time_steps))
#     for i in range(X_ts.shape[0]):
#         X_ts[i,:] = returnIndex[np.arange(i, i + n_lagged_time_steps,1)]
#     r_ts = returnIndex[n_lagged_time_steps-1:-1]    
# =============================================================================
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
# =============================================================================
#     X_ts = np.concatenate((X_ts[nanid:],techid_trun),axis=1)
#     
#     r_ts = r_ts[nanid:]
# =============================================================================
    
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
    
def RRL_Trading_With_Normal_Feature(market_info, 
                                    n_lagged_time_steps, #20 8
                                    learning_rate, 
                                    lambda_re,
                                    delta,
                                    random_state,
                                    n_samples,
                                    window_train,
                                 #   window_test,
                                    n_epochs,
                                    N):

    #  n_lagged_time_steps
    X_ts, r_ts, nanid, p_ts = constructTimeSeriesWithMultiFeatures(market_info, n_samples,  )
    
#    X_ts = preprocessing.MinMaxScaler().fit_transform(X_ts)    
    X_ts = (X_ts - np.mean(X_ts,0))/ np.std(X_ts,0)  # z_score of returns     
#    X_ts = (X_ts - np.mean(X_ts,0))/ np.std(X_ts,0)  # z_score of returns 
    
    pca = PCA(n_components=0.95)
    
    X_reduced = pca.fit_transform(X_ts)
    tmp = X_reduced.copy() ## 
    X_denoised = discrete_wavelet_transform(tmp, 'haar', dec_level=4)
    
    
    # tmp = X_ts.copy()
    # X_denoised = discrete_wavelet_transform(tmp, 'haar', dec_level=4)
    # tmp = X_denoised.copy() ##
    # X_reduced = pca.fit_transform(tmp)
    
    # X_lag, r_lag, nanid_lag = lookbackFeatures(market_info, n_samples, n_lagged_time_steps = X_denoised.shape[1] )  # 
    # X_lag = X_lag[nanid-X_denoised.shape[1]:]
    # X_lag = (X_lag - np.mean(X_lag,0)) / np.std(X_lag,0)
    # r_lag = r_lag[nanid-X_denoised.shape[1]:] 
    
    test = StockTradingStrategy(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)
    
    #   
    listRt_den, listFt_den, listSr_den = test.walkForward(X_denoised, r_ts, window_train, n_epochs, N)    
#    listRt_red, listFt_red, listSr_red = test.walkForward(X_reduced, r_ts, window_train, n_epochs, N)
#    listRt_tec, listFt_tec, listSr_tec = test.walkForward(X_ts, r_ts, window_train, n_epochs, N)  
#    listRt_lag, listFt_lag, listSr_lag = test.walkForward(X_lag, r_lag,  window_train, n_epochs, N)
    
#    listRt_den, listFt_den, listSr_den = test.walkForward(X_denoised, r_ts, p_ts, window_train, n_epochs, N)    
#    listRt_red, listFt_red, listSr_red = test.walkForward(X_reduced, r_ts, p_ts, window_train, n_epochs, N)
#    listRt_tec, listFt_tec, listSr_tec = test.walkForward(X_ts, r_ts, p_ts, window_train, n_epochs, N)  
#    listRt_lag, listFt_lag, listSr_lag = test.walkForward(X_lag, r_lag, p_ts, window_train, n_epochs, N)      
   
    #
    return  listRt_den, listFt_den, listRt_red, listFt_red, #listRt_tec, listRt_lag, nanid      #list_SR


def plot_figure(listRt_den, listFt_den, listRt_red, listRt_tec, listRt_lag, ret, window_train):
     Rs_den = np.asarray(listRt_den)
     Fs_den = np.asarray(listFt_den)
     Rs_red = np.asarray(listRt_red)
     Rs_tec = np.asarray(listRt_tec)
     Rs_lag = np.asarray(listRt_lag)
     
     fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    
     ax[0].plot(market_info['Close'][bah_id:bah_id+len(Rs_den)])
     ax[0].set_ylabel('Price')  #['Close']
     ax[0].set_xlabel('Time')
#     t = np.arange(window_train,window_train+len(Rs1),1)  #9991  truncate series len
            
     ax[1].plot(np.cumsum(Rs_den), 'r', label='Proposed RRL')
     ax[1].plot(np.cumsum(Rs_red), 'b', label='PCA RRL')
     ax[1].plot(np.cumsum(Rs_tec), 'g', label='TA RRL')
     ax[1].plot(np.cumsum(Rs_lag), 'y', label='Basic RRL')
     ax[1].plot(np.cumsum(ret), 'k', label='B&H')
     ax[1].legend(loc="upper left")
     ax[1].set_xlabel('Trading periods')
     ax[1].set_ylabel('Cumulative Profits')

     ax[2].plot(Fs_den[:len(Rs_den)], label='Trading signal')
     ax[2].set_xlabel('Trading periods')
     ax[2].set_ylabel('Signal')
     ax[2].set_ylim(-1.05,1.05)    
     
    
    
random_state = 42 # fixed random seed
#random_state = np.random.randint(100000, size=1).item() # generate random seed
#random_state = 95259

# =============================================================================
# n_lagged_time_steps = 3
# learning_rate = 0.1
# lambda_re = 0.01
# delta = 0.001 #0.005 for sp500, artifi
# n_samples = 6000
# window_train = 30  
# #window_test = 1000 
# n_epochs = 30
# N = 50
# fname = 'data/rb1001.csv' 
# =============================================================================

# =============================================================================
# n_lagged_time_steps = 3
# learning_rate = 0.1
# lambda_re = 0.01
# delta = 0.001 #0.005 for sp500, artifi
# n_samples = 6000
# window_train = 500  
# #window_test = 1000 
# n_epochs = 100
# N = 500
# fname = 'data_old/zcf.csv' # (db2,5)(sym2,5)
# #fname = 'data_old/crude-oil.csv'
# =============================================================================

# =============================================================================
# n_lagged_time_steps = 8
# learning_rate = 0.1
# lambda_re = 0.01
# delta = 0.01 #0.005 for sp500, artifi
# n_samples = 6000

# window_train = 100  # 250
# #window_test = 1000 
# n_epochs = 50
# N = 200
# fname = 'data_old/btc-usd.csv' # 
# =============================================================================

n_lagged_time_steps = 8
learning_rate = 0.1
lambda_re = 0.01
delta = 0.01 #0.005 for sp500, artifi
n_samples = 6000
window_train = 500  
#window_test = 1000 
n_epochs = 100
N = 500

fname = 'data_old/xom.csv'
#fname = 'data_old/hd.csv'
##fname = 'data_old/aapl.csv'

# =============================================================================
# n_lagged_time_steps = 8
# learning_rate = 0.1
# lambda_re = 0.01
# delta = 1 #0.005 for sp500, artifi
# n_samples = 6000
# window_train = 500  
# #window_test = 1000 
# n_epochs = 100
# N = 500
# fname = 'data_old/nyse.csv' # date format
# #fname = "data_old/SP500.csv"
# =============================================================================

# =============================================================================
# n_lagged_time_steps = 8
# learning_rate = 0.1
# lambda_re = 0.01
# delta = 1 #0.005 for sp500, artifi
# n_samples = 6000
# window_train = 500  
# #window_test = 1000 
# n_epochs = 100
# N = 500
# fname = 'data_old/szzs.csv'
# ##fname = 'data_old/nasdaq.csv'
# =============================================================================

# =============================================================================

# n_lagged_time_steps = 8
# learning_rate = 0.1
# lambda_re = 0.01
# delta = 0.01 #0.005 for sp500, artifi
# n_samples = 6000
# window_train = 500  
# #window_test = 1000 
# n_epochs = 100
# N = 200
# fname = "data_old/gzmt.csv"
# =============================================================================
## N=200, window_train=500 or N=500, window_train=1000 or N=1000, window_train=1000

# fname = 'data/z3.csv'   
# df = pd.read_csv(fname, header=None)
# forex_market_info = df[0] #df[0]

#plt.plot(forex_market_info[window_train:window_train+window_test])
#plt.xlabel('t')
#plt.ylabel('Price')
#plt.grid()
#plt.show()

df = pd.read_csv(fname, header=None)

time_str = df[0] #+" " + df[1]
#dates = [dt.strptime(time_str[i], '%Y/%m/%d %H:%M') for i in range(len(time_str))]
#dates = [dt.strptime(time_str[i], '%Y/%m/%d') for i in range(len(time_str))]
dates = [dt.strptime(i, '%Y-%m-%d') for i in time_str]
market_info = df.iloc[:,1:]
market_info.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
market_info.index = dates

#ret = market_info.Close.astype(np.float64).diff().values
ret = market_info['Close'].diff().values
### 
# listRt_tec, listRt_lag, nanid 
listRt_den, listFt_den, listRt_red, listFt_red,  = RRL_Trading_With_Normal_Feature(market_info, 
                                                            n_lagged_time_steps,
                                                            learning_rate,
                                                            lambda_re,
                                                            delta,
                                                            random_state,
                                                            n_samples,
                                                            window_train,
                                                          #  window_test,
                                                            n_epochs,
                                                            N)  
#                   #    list_SR  strs='sp500,8,0.1,0.01,0.01,42,6000,500,150,500'

                   #print('Testing Set Sharpe Ratio : {:.4f}'.format(SR1))
print('Our Cumulative Profit : {:.2f}'.format(np.sum(listRt_den)))
print('Pca Cumulative Profit : {:.2f}'.format(np.sum(listRt_red)))
print('Tec Cumulative Profit : {:.2f}'.format(np.sum(listRt_tec)))
print('Lag Cumulative Profit : {:.2f}'.format(np.sum(listRt_lag)))
bah_id = nanid + window_train
print('Bah Cumulative Profit : {:.2f}'.format(np.sum(ret[bah_id:bah_id+len(listRt_den)])))

tf = [listFt_den[i-1]==listFt_den[i] for i in np.arange(1,len(listFt_den)) ]
print(tf.count(0)) 

plot_figure(listRt_den, listFt_den, listRt_red, listRt_tec, listRt_lag, ret[bah_id:bah_id+len(listRt_den)], window_train )  # ['Close']

#plot_fig(listRt_den)

### calculate annualized rate of return
# =============================================================================
# ror_den = np.sum(listRt_den) / market_info['Close'][bah_id] * np.sqrt(252)
# ror_red = np.sum(listRt_red) / market_info['Close'][bah_id] * np.sqrt(252)
# ror_tec = np.sum(listRt_tec) / market_info['Close'][bah_id] * np.sqrt(252)
# ror_lag = np.sum(listRt_lag) / market_info['Close'][bah_id] * np.sqrt(252)
# ror_bah = np.sum(ret[bah_id:bah_id+len(listRt_den)]) / market_info['Close'][bah_id] * np.sqrt(252)
# =============================================================================

ror_den = np.power((np.sum(listRt_den)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_den))-1
if np.sum(listRt_red) >= 0 :
    ror_red = np.power((np.sum(listRt_red)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_red))-1
else:
    ror_red = -np.power((-np.sum(listRt_red)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_red))-1
if np.sum(listRt_tec) >= 0:
    ror_tec = np.power((np.sum(listRt_tec)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_tec))-1
else:
    ror_tec = -np.power((-np.sum(listRt_tec)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_tec))-1
if np.sum(listRt_lag) >= 0:
    ror_lag = np.power((np.sum(listRt_lag)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_lag))-1
else:
    ror_lag = -np.power((-np.sum(listRt_lag)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_lag))-1
if np.sum(ret[bah_id:bah_id+len(listRt_den)]) >= 0:
    ror_bah = np.power((np.sum(ret[bah_id:bah_id+len(listRt_den)])+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_den))-1
else:
    ror_bah = -np.power((-np.sum(ret[bah_id:bah_id+len(listRt_den)])+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_den))-1


### calculate annualized sharpe ratio
sr_den = np.mean(listRt_den) / np.std(listRt_den) * np.sqrt(252)
sr_red = np.mean(listRt_red) / np.std(listRt_red) * np.sqrt(252)
sr_tec = np.mean(listRt_tec) / np.std(listRt_tec) * np.sqrt(252)
sr_lag = np.mean(listRt_lag) / np.std(listRt_lag) * np.sqrt(252)
sr_bah = np.mean(ret[bah_id:bah_id+len(listRt_den)]) / np.std(ret[bah_id:bah_id+len(listRt_den)])* np.sqrt(252)

### calculate mdd
den = np.cumsum(listRt_den)
trough_index = np.argmax(np.maximum.accumulate(den) - den)
peak_index = np.argmax(den[:trough_index])
mdd_den = (den[peak_index] - den[trough_index])/ den[peak_index]


red = np.cumsum(listRt_red)
trough_index = np.argmax(np.maximum.accumulate(red) - red)
peak_index = np.argmax(red[:trough_index])
mdd_red = (red[peak_index] - red[trough_index])/ red[peak_index] 

tec = np.cumsum(listRt_tec)
trough_index = np.argmax(np.maximum.accumulate(tec) - tec)
peak_index = np.argmax(tec[:trough_index])
mdd_tec = (tec[peak_index] - tec[trough_index])/ tec[peak_index]

lag = np.cumsum(listRt_lag)
trough_index = np.argmax(np.maximum.accumulate(lag) - lag)
peak_index = np.argmax(lag[:trough_index])
mdd_lag = (lag[peak_index] - lag[trough_index])/ lag[peak_index]

bah = np.cumsum(ret[bah_id:bah_id+len(listRt_den)])
trough_index = np.argmax(np.maximum.accumulate(bah) - bah)
peak_index = np.argmax(bah[:trough_index])
mdd_bah = (bah[peak_index] - bah[trough_index])/ bah[peak_index]

### calculate calmar ratio
cr_den = ror_den / mdd_den
cr_red = ror_red / mdd_red
cr_tec = ror_tec / mdd_tec
cr_lag = ror_lag / mdd_lag
cr_bah = ror_bah / mdd_bah

### calculate annualized sortino ratio 
ddsig_den = np.std([i for i in listRt_den if i < 0.])
ddsig_red = np.std([i for i in listRt_red if i < 0.])
ddsig_tec = np.std([i for i in listRt_tec if i < 0.])
ddsig_lag = np.std([i for i in listRt_lag if i < 0.])
ddsig_bah = np.std([i for i in ret[bah_id:bah_id+len(listRt_den)] if i < 0.])

sro_den = np.mean(listRt_den) / ddsig_den * np.sqrt(252)
sro_red = np.mean(listRt_red) / ddsig_red * np.sqrt(252)
sro_tec = np.mean(listRt_tec) / ddsig_tec * np.sqrt(252)
sro_lag = np.mean(listRt_lag) / ddsig_lag * np.sqrt(252)
sro_bah = np.mean(ret[bah_id:bah_id+len(listRt_den)]) / ddsig_bah *np.sqrt(252)

print('ror:',round(ror_den,2),'\t', round(ror_red,2),'\t',round(ror_tec,2),'\t',round(ror_lag,2),'\t', round(ror_bah,2), '\n', 
      'sr:', round(sr_den,2),'\t', round(sr_red,2),'\t', round(sr_tec,2),'\t', round(sr_lag,2),'\t', round(sr_bah,2), '\n', 
      'mdd:',round(mdd_den,2), '\t', round(mdd_red,2), '\t',round(mdd_tec,2), '\t',round(mdd_lag,2), '\t', round(mdd_bah,2), '\n',
      'cr:',round(cr_den,2), '\t',round(cr_red,2), '\t',round(cr_tec,2), '\t',round(cr_lag,2), '\t', round(cr_bah,2), '\n',
      'sro:', round(sro_den,2), '\t',round(sro_red,2), '\t',round(sro_tec,2), '\t',round(sro_lag,2), '\t', round(sro_bah,2))

