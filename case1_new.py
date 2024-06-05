# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:42:46 2021

@author: de
"""

### compare PCA&DWT RRL, TI RRL, B&H, ARIMA

import numpy as np
import pandas as pd
from models.Molina import  StockTradingStrategy as StockTradingStrategy1
from models.MolinaFuture import StockTradingStrategy as StockTradingStrategy2
from sklearn.model_selection import ParameterGrid
#from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime as dt


from sklearn.decomposition import PCA
import pywt
#from scipy.signal import fftconvolve
import scipy.io
import talib as ta


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


def tradingPcadwt1(market_info, 
                 learning_rate, 
                 lambda_re,
                 delta,
                 random_state,
                 n_samples,
                 window_train,
                 n_epochs,
                 N):

    X_ts, r_ts, nanid, p_ts = constructTimeSeriesWithMultiFeatures(market_info, n_samples,  )
    
    X_ts = (X_ts - np.mean(X_ts,0))/ np.std(X_ts,0)  # z_score of returns     
    
    pca = PCA(n_components=0.95)
    
    X_reduced = pca.fit_transform(X_ts)
    tmp = X_reduced.copy() ## 
    X_denoised = discrete_wavelet_transform(tmp, 'haar', dec_level=4)
    

    test = StockTradingStrategy1(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)    
### Proposed RRL               
    
#    listRt_den, listFt_den, listSr_den = test.walkForward(X_denoised, r_ts, p_ts, window_train, n_epochs, N) 
    listRt_den, listFt_den, listSr_den = test.walkForward(X_denoised, r_ts, window_train, n_epochs, N)                    
   
    return  listRt_den, listFt_den,  nanid    

def tradingTec1(market_info, 
                 learning_rate, 
                 lambda_re,
                 delta,
                 random_state,
                 n_samples,
                 window_train,
                 n_epochs,
                 N):
    
    X_ts, r_ts, nanid, p_ts = constructTimeSeriesWithMultiFeatures(market_info, n_samples,  ) 

    X_ts = (X_ts - np.mean(X_ts,0))/ np.std(X_ts,0)  # z_score of returns 
    
    test = StockTradingStrategy1(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)

#    listRt_tec, listFt_tec, listSr_tec = test.walkForward(X_ts, r_ts, p_ts, window_train, n_epochs, N)   
    
    listRt_tec, listFt_tec, listSr_tec = test.walkForward(X_ts, r_ts, window_train, n_epochs, N)     
    
    return listRt_tec, listFt_tec
    
def tradingPcadwt2(market_info, 
                 learning_rate, 
                 lambda_re,
                 delta,
                 random_state,
                 n_samples,
                 window_train,
                 n_epochs,
                 N):

    X_ts, r_ts, nanid, p_ts = constructTimeSeriesWithMultiFeatures(market_info, n_samples,  )
    
    X_ts = (X_ts - np.mean(X_ts,0))/ np.std(X_ts,0)  # z_score of returns     
    
    pca = PCA(n_components=0.95)
    
    X_reduced = pca.fit_transform(X_ts)
    tmp = X_reduced.copy() ## 
    X_denoised = discrete_wavelet_transform(tmp, 'haar', dec_level=4)
    

    test = StockTradingStrategy2(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)    
### Proposed RRL               
    
    listRt_den, listFt_den, listSr_den = test.walkForward(X_denoised, r_ts, p_ts, window_train, n_epochs, N) 
#    listRt_den, listFt_den, listSr_den = test.walkForward(X_denoised, r_ts, window_train, n_epochs, N)                    
   
    return  listRt_den, listFt_den,  nanid    

def tradingTec2(market_info, 
                 learning_rate, 
                 lambda_re,
                 delta,
                 random_state,
                 n_samples,
                 window_train,
                 n_epochs,
                 N):
    
    X_ts, r_ts, nanid, p_ts = constructTimeSeriesWithMultiFeatures(market_info, n_samples,  ) 

    X_ts = (X_ts - np.mean(X_ts,0))/ np.std(X_ts,0)  # z_score of returns 
    
    test = StockTradingStrategy2(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)

    listRt_tec, listFt_tec, listSr_tec = test.walkForward(X_ts, r_ts, p_ts, window_train, n_epochs, N)   
    
#    listRt_tec, listFt_tec, listSr_tec = test.walkForward(X_ts, r_ts, window_train, n_epochs, N)     
    
    return listRt_tec, listFt_tec

def plot_figure(listRt_den, listFt_den, listRt_tec, arima, ret):
     Rs_den = np.asarray(listRt_den)
     Fs_den = np.asarray(listFt_den)
     Rs_lag = np.asarray(listRt_tec)
#    Fs_lag = np.asarray(listFt_lag)
     
     fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
    
     ax[0].plot(market_info['Close'][bah_id:bah_id+len(Rs_den)])
     ax[0].set_ylabel('Price ($)')  #['Close']
     ax[0].set_xlabel('Time')
#     t = np.arange(window_train,window_train+len(Rs1),1)
            
     ax[1].plot(np.cumsum(ret), 'k', label='B&H')
     ax[1].plot(np.cumsum(arima), 'b', label='ARIMA')
     ax[1].plot(np.cumsum(Rs_lag), 'y', label='TI RRL')
     ax[1].plot(np.cumsum(Rs_den), 'r', label='PCA&DWT RRL')
     ax[1].legend(loc="upper left")
     ax[1].set_xlabel('Trading periods')
     ax[1].set_ylabel('Cumulative Profits ($)')

     ax[2].plot(Fs_den[:len(Rs_den)])
     ax[2].set_xlabel('Trading periods')
     ax[2].set_ylabel('Trading Signals of PCA&DWT RRL')
     ax[2].set_ylim(-1.05,1.05)        
     
     plt.savefig('', kwargs)
# plt.annotate('focus',xy=(-5,0),xytext=(-2,0.25),arrowprops = dict(facecolor='red',shrink=0.05,headlength= 20,headwidth = 20))
     # ax[3].plot(Fs_lag[:len(Rs_lag)])
     # ax[3].set_xlabel('Trading periods')
     # ax[3].set_ylabel('Signal of Basic RRL')
     # ax[3].set_ylim(-1.05,1.05)   
#ax.annotate('local max', xy=(2,1), xytext=(3,1.5),
#            arrowprops=dict(facecolor='black', shrink=0.05))
# xy=(横坐标，纵坐标)  箭头尖端
# xytext=(横坐标，纵坐标) 文字的坐标，指的是最左边的坐标
# arrowprops= {facecolor= '颜色', shrink = '数字' <1  收缩箭头}

def plot_all(dp_all):
     nyse = dp_all[0]
     xom = dp_all[1]
     zcf = dp_all[2]
     
     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(21,12))
             
     ax[0,0].plot(nyse[0])
     ax[0,0].set_ylabel('Price ($)')  #['Close']
     ax[0,0].set_xlabel('Time')

     ax[0,1].plot(xom[0])
     ax[0,1].set_ylabel('Price ($)')  #['Close']
     ax[0,1].set_xlabel('Time')

     ax[0,2].plot(zcf[0])
     ax[0,2].set_ylabel('Price ($)')  #['Close']
     ax[0,2].set_xlabel('Time')
            
     ax[1,0].plot(np.cumsum(nyse[1]), 'k', label='B&H')
     ax[1,0].plot(np.cumsum(nyse[2]), 'b', label='ARIMA')
     ax[1,0].plot(np.cumsum(nyse[3]), 'y', label='TI RRL')
     ax[1,0].plot(np.cumsum(nyse[4]), 'r', label='PCA&DWT RRL')
     ax[1,0].legend(loc="upper left", borderpad=0.1)
     ax[1,0].set_xlabel('Trading Periods')
     ax[1,0].set_ylabel('Cumulative Profits ($)')

     ax[1,1].plot(np.cumsum(xom[1]), 'k', label='B&H')
     ax[1,1].plot(np.cumsum(xom[2]), 'b', label='ARIMA')
     ax[1,1].plot(np.cumsum(xom[3]), 'y', label='TI RRL')
     ax[1,1].plot(np.cumsum(xom[4]), 'r', label='PCA&DWT RRL')
     ax[1,1].legend(loc="upper left", borderpad=0.1)
     ax[1,1].set_xlabel('Trading Periods')
     ax[1,1].set_ylabel('Cumulative Profits ($)')

     ax[1,2].plot(np.cumsum(zcf[1]), 'k', label='B&H')
     ax[1,2].plot(np.cumsum(zcf[2]), 'b', label='ARIMA')
     ax[1,2].plot(np.cumsum(zcf[3]), 'y', label='TI RRL')
     ax[1,2].plot(np.cumsum(zcf[4]), 'r', label='PCA&DWT RRL')
     ax[1,2].legend(loc="upper left", borderpad=0.1)
     ax[1,2].set_xlabel('Trading Periods')
     ax[1,2].set_ylabel('Cumulative Profits ($)')
     
     ax[2,0].plot(nyse[-1])
     ax[2,0].set_xlabel('Trading Periods')
     ax[2,0].set_ylabel('Trading Signals of PCA&DWT RRL')
     ax[2,0].set_ylim(-1.05,1.05)        

     ax[2,1].plot(nyse[-1])
     ax[2,1].set_xlabel('Trading Periods')
     ax[2,1].set_ylabel('Trading Signals of PCA&DWT RRL')
     ax[2,1].set_ylim(-1.05,1.05)  
     
     ax[2,2].plot(nyse[-1])
     ax[2,2].set_xlabel('Trading Periods')
     ax[2,2].set_ylabel('Trading Signals of PCA&DWT RRL')
     ax[2,2].set_ylim(-1.05,1.05)       
     
     plt.savefig('1all.eps',dpi=600, bbox_inches='tight')
#random_state = 96211 #42 # fixed random seed
#random_state = np.random.randint(100000, size=1).item() # generate random seed
#random_state = 87142,96211


# #fname = 'data_old/zc=f.csv' # (db2,5)(sym2,5)
# fname = 'data_old/crude-oil.csv'
# param_basic = { # zcf: 15443, 9, 83468, 9 clf: 76022, 8
# 'random_state':  [76022], #[np.random.randint(100000, size=1).item() for _ in range(20)], 
# 'n_lagged': [8],
# 'learning_rate': [0.1],
# 'lambda_re':  [0.01],
# 'delta': [0.001], #0.005 for sp500, artifi
# 'n_samples': [6000],
# 'window_train': [500],  
# 'n_epochs': [100],
# 'N':  [500]
# }         


# =============================================================================
# #fname = 'data_old/szzs.csv'
# #fname = 'data_old/nyse.csv'
# param_basic = {  # szzs: 154, 12  nyse: 37834, 9
# 'random_state': [np.random.randint(100000, size=1).item() for _ in range(10)], # 36229
# 'n_lagged': [8,9],
# 'learning_rate': [0.1],
# 'lambda_re':  [0.01],
# 'delta': [1], #0.005 for sp500, artifi
# 'n_samples': [6000],
# 'window_train': [500],  
# 'n_epochs': [100],
# 'N':  [500]
# }
# =============================================================================

fnames = ['data_old/nyse.csv', 'data_old/xom.csv', 'data_old/zcf.csv']
dp_all = []

#fname = 'data_old/nyse.csv'
for fname in fnames:

    if fname == 'data_old/nyse.csv':
        tc = 1
    elif fname == 'data_old/xom.csv':
        tc = 0.01
    else:
        tc = 0.001   
        
    param_pcadwt = { 'random_state': [42], 'learning_rate': [0.1],'lambda_re':  [0.01], 'delta': [tc], 
    'n_samples': [6000], 'window_train': [500], 'n_epochs': [100], 'N':  [500] }
    
    path = 'D:/A_Mannheim/Quanttrade/data/arima_' + fname[fname.find('/')+1:fname.find('.')] + '.mat'
    
    arima = scipy.io.loadmat(path)['data'].flatten() 

# fname = 'data_old/xom.csv'
# param_pcadwt = { 
# 'random_state': [42], 
# 'learning_rate': [0.1],
# 'lambda_re':  [0.01],
# 'delta': [0.01], 
# 'n_samples': [6000],
# 'window_train': [500],  
# 'n_epochs': [100],
# 'N':  [500]
# }     
#arima = scipy.io.loadmat('D:/A_Mannheim/Quanttrade/data/arima_xom.mat')['data'].flatten() 

# fname = 'data_old/zcf.csv'
# param_pcadwt = { 
# 'random_state': [42], 
# 'learning_rate': [0.1],
# 'lambda_re':  [0.01],
# 'delta': [0.001], 
# 'n_samples': [6000],
# 'window_train': [500],  
# 'n_epochs': [100],
# 'N':  [500]
# }     
#arima = scipy.io.loadmat('D:/A_Mannheim/Quanttrade/data/arima_zcf.mat')['data'].flatten() 
    df = pd.read_csv(fname, header=None)

    time_str = df[0] #+" " + df[1]
#dates = [dt.strptime(time_str[i], '%Y/%m/%d %H:%M') for i in range(len(time_str))]
#dates = [dt.strptime(time_str[i], '%Y/%m/%d') for i in range(len(time_str))]
    dates = [dt.strptime(time_str[i], '%Y-%m-%d') for i in range(len(time_str))]
    market_info = df.iloc[:,1:]
    market_info.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    market_info.index = dates
#pt_close = market_info['Close']
    ret = market_info['Close'].diff().values
    
    nanid = 63    
    window_train = 500
    bah_id = nanid + window_train
    
#    dp = []
    if fname != 'data_old/zcf.csv':
        
         for it in ParameterGrid(param_pcadwt):
           
             listRt_den, listFt_den, nanid  = tradingPcadwt1(market_info, **it)  

             listRt_tec, listFt_tec = tradingTec1(market_info, **it)
             # print(63==nanid)
             # dp.append(market_info['Close'][bah_id:bah_id+len(listRt_den)])
             # dp.append(ret[bah_id:bah_id+len(listRt_den)])
             # dp.append(arima)
             # dp.append(listRt_tec)
             # dp.append(listRt_den)
             # dp.append(listFt_den)
             
    else:
         for it in ParameterGrid(param_pcadwt):
           
             listRt_den, listFt_den, nanid  = tradingPcadwt2(market_info, **it)  

             listRt_tec, listFt_tec = tradingTec2(market_info, **it)         
             # print(63==nanid)
             # dp.append(market_info['Close'][bah_id:bah_id+len(listRt_den)])
             # dp.append(ret[bah_id:bah_id+len(listRt_den)])
             # dp.append(arima)
             # dp.append(listRt_tec)
             # dp.append(listRt_den)
             # dp.append(listFt_den)             

#    dp_all.append(dp)

#plot_all(dp_all)    


    tf = [listFt_den[i-1]==listFt_den[i] for i in np.arange(1,len(listFt_den)) ]
    print(tf.count(0)) 

    plot_figure(listRt_den, listFt_den, listRt_tec, arima, ret[bah_id:bah_id+len(listRt_den)])      

    np_den = np.sum(listRt_den)
    np_tec = np.sum(listRt_tec)
    np_ama = np.sum(arima)
    np_bah = np.sum(ret[bah_id:bah_id+len(listRt_den)])

    apy_den = np.power((np.sum(listRt_den)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_den))-1

    if np.sum(listRt_tec) >= 0:
         apy_tec = np.power((np.sum(listRt_tec)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_tec))-1
    else:
         apy_tec = -np.power((-np.sum(listRt_tec)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_tec))-1
    if np.sum(arima) >= 0:
         apy_ama = np.power((np.sum(arima)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(arima))-1
    else:
         apy_ama = -np.power((-np.sum(arima)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(arima))-1
    if np.sum(ret[bah_id:bah_id+len(listRt_den)]) >= 0:
         apy_bah = np.power((np.sum(ret[bah_id:bah_id+len(listRt_den)])+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_den))-1
    else:
         apy_bah = -np.power((-np.sum(ret[bah_id:bah_id+len(listRt_den)])+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_den))-1
    
### calculate annualized sharpe ratio
    sr_den = (np.mean(listRt_den) - 0.01) / np.std(listRt_den) * np.sqrt(252)   ### apy_den
    sr_tec = (np.mean(listRt_tec) - 0.01) / np.std(listRt_tec) * np.sqrt(252)
    sr_ama = (np.mean(arima) - 0.01) / np.std(arima) * np.sqrt(252) 
    sr_bah = (np.mean(ret[bah_id:bah_id+len(listRt_den)]) - 0.01) / np.std(ret[bah_id:bah_id+len(listRt_den)])* np.sqrt(252)    

### calculate mdd
    den = np.cumsum(listRt_den)
    trough_index = np.argmax(np.maximum.accumulate(den) - den)
    peak_index = np.argmax(den[:trough_index])
    mdd_den = (den[peak_index] - den[trough_index])/ den[peak_index]

    tec = np.cumsum(listRt_tec)
    trough_index = np.argmax(np.maximum.accumulate(tec) - tec)
    peak_index = np.argmax(tec[:trough_index])
    mdd_tec = (tec[peak_index] - tec[trough_index])/ tec[peak_index]

    ama = np.cumsum(arima)
    trough_index = np.argmax(np.maximum.accumulate(ama) - ama)
    peak_index = np.argmax(ama[:trough_index])
    mdd_ama = (ama[peak_index] - ama[trough_index])/ ama[peak_index]

    bah = np.cumsum(ret[bah_id:bah_id+len(listRt_den)])
    trough_index = np.argmax(np.maximum.accumulate(bah) - bah)
    peak_index = np.argmax(bah[:trough_index])
    mdd_bah = (bah[peak_index] - bah[trough_index])/ bah[peak_index]

### calculate calmar ratio
    cr_den = apy_den / mdd_den
    cr_tec = apy_tec / mdd_tec
    cr_ama = apy_ama / mdd_ama
    cr_bah = apy_bah / mdd_bah

### calculate annualized sortino ratio 
# ddsig_den = np.std([i for i in listRt_den if i < 0.])
# ddsig_tec = np.std([i for i in listRt_tec if i < 0.])
# ddsig_bah = np.std([i for i in ret[bah_id:bah_id+len(listRt_den)] if i < 0.])

# sro_den = np.mean(listRt_den) / ddsig_den * np.sqrt(252)
# sro_lag = np.mean(listRt_lag) / ddsig_lag * np.sqrt(252)
# sro_bah = np.mean(ret[bah_id:bah_id+len(listRt_den)]) / ddsig_bah *np.sqrt(252)

    print('np:', round(np_den,2),'\t', round(np_tec,2),'\t', round(np_ama,2),'\t',round(np_bah,2),'\n',
      'apy:',round(apy_den,2),'\t', round(apy_tec,2),'\t', round(apy_ama,2),'\t', round(apy_bah,2), '\n', 
      'sr:', round(sr_den,2),'\t', round(sr_tec,2),'\t', round(sr_ama,2),'\t', round(sr_bah,2), '\n', 
      'mdd:',round(mdd_den,2), '\t', round(mdd_tec,2), '\t', round(mdd_ama,2),'\t', round(mdd_bah,2), '\n',
      'cr:',round(cr_den,2), '\t', round(cr_tec,2), '\t', round(cr_ama,2),'\t', round(cr_bah,2), '\n',)

##'sro:', round(sro_den,2), '\t', round(sro_lag,2), '\t', round(sro_bah,2)

# print('PCADWT RRL CP : {:.2f}'.format(np.sum(listRt_den)))
# print('TI RRL CP : {:.2f}'.format(np.sum(listRt_tec)))
# print('ARIMA CP : {:.2f}'.format(np.sum(arima)))
# print('BAH CP : {:.2f}'.format(np.sum(ret[bah_id:bah_id+len(listRt_den)])))


## N=200, window_train=500 or N=500, window_train=1000 or N=1000, window_train=1000

# fname = 'data/z3.csv'   
# df = pd.read_csv(fname, header=None)
# forex_market_info = df[0] #df[0]

#plt.plot(forex_market_info[window_train:window_train+window_test])
#plt.xlabel('t')
#plt.ylabel('Price')
#plt.grid()
#plt.show()