# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 20:53:23 2021

@author: de
"""
### compare PCADWT RRL, DWT RRL, PCA RRL, TI RRL



import numpy as np
import pandas as pd
from models.Molina import  StockTradingStrategy
#from models.MolinaFuture import StockTradingStrategy
from sklearn.model_selection import ParameterGrid
#from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime as dt


from sklearn.decomposition import PCA
import pywt
#from scipy.signal import fftconvolve
#import scipy.io
import talib as ta

import time 

start = time.process_time()
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

def tradingPcadwt(market_info, 
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
    

    test = StockTradingStrategy(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)    
###  B&H, Basic RRL,  Proposed RRL
# =============================================================================
#     listRt_den, listFt_den, listSr_den = test.walkForward(X_denoised, r_ts, p_ts, window_train, n_epochs, N)                     
#     listRt_lag, listFt_lag, listSr_lag = test.walkForward(X_lag, r_lag, p_ts, window_train, n_epochs, N)      
# =============================================================================
    
#    listRt_den, listFt_den, listSr_den = test.walkForward(X_denoised, r_ts, p_ts, window_train, n_epochs, N) 
    listRt_den, listFt_den, listSr_den = test.walkForward(X_denoised, r_ts, window_train, n_epochs, N)                    
    
    return  listRt_den, listFt_den,  nanid    

def tradingDwt(market_info, 
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
    
    tmp = X_ts.copy() ## 
    X_dwt = discrete_wavelet_transform(tmp, 'haar', dec_level=4)
    
    test = StockTradingStrategy(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)
          
#    listRt_dwt, listFt_dwt, listSr_dwt = test.walkForward(X_dwt, r_ts, p_ts, window_train, n_epochs, N) 
    listRt_dwt, listFt_dwt, listSr_dwt = test.walkForward(X_dwt, r_ts, window_train, n_epochs, N)
      
    return   listRt_dwt, listFt_dwt, nanid #listRt_den, listFt_den, nanid 

def tradingPca(market_info, 
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
    
    X_pca = pca.fit_transform(X_ts)
    
    test = StockTradingStrategy(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)

#    listRt_pca, listFt_pca, listSr_pca = test.walkForward(X_pca, r_ts, p_ts, window_train, n_epochs, N)                   
    
    listRt_pca, listFt_pca, listSr_pca = test.walkForward(X_pca, r_ts, window_train, n_epochs, N)
      
    return   listRt_pca, listFt_pca, nanid #listRt_den, listFt_den, nanid 

def tradingTec(market_info, 
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
    
    test = StockTradingStrategy(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)

#    listRt_tec, listFt_tec, listSr_tec = test.walkForward(X_ts, r_ts, p_ts, window_train, n_epochs, N)   
    
    listRt_tec, listFt_tec, listSr_tec = test.walkForward(X_ts, r_ts, window_train, n_epochs, N)     
    
    return listRt_tec, listFt_tec

def plot_figure(nyse, xom, zcf):

      fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
       
          
      ax[0].plot(np.cumsum(nyse[-1]), 'y', label='TI RRL')
      ax[0].plot(np.cumsum(nyse[-2]), 'b', label='PCA RRL')
      ax[0].plot(np.cumsum(nyse[1]), 'k', label='DWT RRL')
      ax[0].plot(np.cumsum(nyse[0]), 'r', label='PCA&DWT RRL')
      ax[0].legend(loc="upper left")
      ax[0].set_xlabel('Trading  Periods')
      ax[0].set_ylabel('Cumulative Profits ($)')
      
      ax[1].plot(np.cumsum(xom[-1]), 'y', label='TI RRL')
      ax[1].plot(np.cumsum(xom[-2]), 'b', label='PCA RRL')
      ax[1].plot(np.cumsum(xom[1]), 'k', label='DWT RRL')
      ax[1].plot(np.cumsum(xom[0]), 'r', label='PCA&DWT RRL')
      ax[1].legend(loc="upper left")
      ax[1].set_xlabel('Trading  Periods')
      ax[1].set_ylabel('Cumulative Profits ($)')    
      
      ax[2].plot(np.cumsum(zcf[-1]), 'y', label='TI RRL')
      ax[2].plot(np.cumsum(zcf[-2]), 'b', label='PCA RRL')
      ax[2].plot(np.cumsum(zcf[1]), 'k', label='DWT RRL')
      ax[2].plot(np.cumsum(zcf[0]), 'r', label='PCA&DWT RRL')
     
      ax[2].legend(loc="upper left")
      ax[2].set_xlabel('Trading  Periods')
      ax[2].set_ylabel('Cumulative Profits ($)')
      
     # den_cps = np.sum(den_all, axis=1)
     # red_cps = np.sum(red_all, axis=1)
     # tec_cps = np.sum(tec_all, axis=1)
     # cps = [tec_cps, red_cps, den_cps]
     # ax[1].boxplot(cps, labels=['TA RRL', 'PCA RRL', 'PCA&DWT RRL'], notch=True, sym='rx', vert=True)
     # ax[1].set_xlabel('Trading Strategies')
     # ax[1].set_ylabel('Cumulative Profits')
     
     # ax[1].plot(Fs_den[:len(Rs_den)])
     # ax[1].set_xlabel('Trading periods')
     # ax[1].set_ylabel('Signal of PCA&DWT RRL')
     # ax[1].set_ylim(-1.05,1.05)   
     
     # ax[2].plot(Fs_red[:len(Rs_red)])
     # ax[2].set_xlabel('Trading periods')
     # ax[2].set_ylabel('Signal of PCA RRL')
     # ax[2].set_ylim(-1.05,1.05) 

     # ax[3].plot(Fs_tec[:len(Rs_tec)])
     # ax[3].set_xlabel('Trading periods')
     # ax[3].set_ylabel('Signal of TA RRL')
     # ax[3].set_ylim(-1.05,1.05)
     
#random_state = [np.random.randint(100000, size=1).item() for _ in range(20)]      

fname = 'data_old/nyse.csv'
#fname = 'data_old/xom.csv'
#fname = 'data_old/zcf.csv'  

param_pcadwt = {
'random_state': [42],  #random_state,
'learning_rate': [0.1],
'lambda_re':  [0.01],
'delta': [1], #0.001
'n_samples': [6000],
'window_train': [500],  
'n_epochs': [100],
'N':  [500]
}

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

### 
#den_all = []
#dwt_all = []
#pca_all = []
#tec_all = []

#nyse_all = []
#xom_all = []
#zcf_all = []

for it in ParameterGrid(param_pcadwt):
    listRt_den, listFt_den, nanid  = tradingPcadwt(market_info, **it)  
#    print(it['random_state'], np.sum(listRt_den))
#    den_all.append(listRt_den)   
    end = time.process_time()
    print('running time:%s'%(end-start))
    
#    nyse_all.append(listRt_den)
#    xom_all.append(listRt_den)
#    zcf_all.append(listRt_den)
    
    listRt_dwt, listFt_dwt, nanid  = tradingDwt(market_info, **it)  
#    print(it['random_state'], np.sum(listRt_dwt))
#    dwt_all.append(listRt_dwt) 
#    xom_all.append(listRt_dwt)
#    zcf_all.append(listRt_dwt)
    
    listRt_pca, listFt_pca, nanid  = tradingPca(market_info, **it)
#    print(it['random_state'], np.sum(listRt_pca))
#    pca_all.append(listRt_pca)
#    xom_all.append(listRt_pca)
#    zcf_all.append(listRt_pca)
    
    listRt_tec, listFt_tec = tradingTec(market_info, **it)
#    print(it['random_state'], np.sum(listRt_tec))
#    tec_all.append(listRt_tec)
#    xom_all.append(listRt_tec)
#    zcf_all.append(listRt_tec)
    

#np.save('D:/A_Mannheim/Quanttrade/results_new/xom2_all.npy', np.array(xom_all))
#np.save('D:/A_Mannheim/Quanttrade/results_new/zcf2_all.npy', np.array(zcf_all))

#np.save('D:/A_Mannheim/Quanttrade/data/t2den_clf.npy', np.array(den_all))

#np.save('D:/A_Mannheim/Quanttrade/data/t2tec_clf.npy', np.array(tec_all))

# den_all = np.load('D:/A_Mannheim/Quanttrade/data/t2den_szzs.npy')
# red_all = np.load('D:/A_Mannheim/Quanttrade/data/t2red_szzs.npy')
# tec_all = np.load('D:/A_Mannheim/Quanttrade/data/t2tec_szzs.npy')
#nyse_all = np.load('D:/A_Mannheim/Quanttrade/results_new/nyse2_all.npy')
#xom_all = np.load('D:/A_Mannheim/Quanttrade/results_new/xom2_all.npy')
#zcf_all = np.load('D:/A_Mannheim/Quanttrade/results_new/zcf2_all.npy')

#listRt_den = np.mean(den_all, axis=0)

#listRt_tec = np.mean(tec_all, axis=0)

#nanid = 63    
window_train = 500

bah_id = nanid + window_train
#print('Bah Cumulative Profit : {:.2f}'.format(np.sum(ret[bah_id:bah_id+len(listRt_red)])))

#plot_figure(listRt_den, listRt_dwt, listRt_pca,  listRt_tec)      
#plot_figure(nyse_all, xom_all, zcf_all,) 

np_den = np.sum(listRt_den)
np_dwt = np.sum(listRt_dwt)
np_pca = np.sum(listRt_pca)
np_tec = np.sum(listRt_tec)

if np.sum(listRt_den) >= 0:
    ror_den = np.power((np.sum(listRt_den)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_den))-1
else:
    ror_den = -np.power((-np.sum(listRt_den)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_den))-1    

if np.sum(listRt_dwt) >= 0:
    ror_dwt = np.power((np.sum(listRt_dwt)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_dwt))-1
else:
    ror_dwt = -np.power((-np.sum(listRt_dwt)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_dwt))-1
    
if np.sum(listRt_pca) >= 0:
    ror_pca = np.power((np.sum(listRt_pca)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_pca))-1
else:
    ror_pca = -np.power((-np.sum(listRt_pca)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_pca))-1
    
if np.sum(listRt_tec) >= 0:
    ror_tec = np.power((np.sum(listRt_tec)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_tec))-1
else:
    ror_tec = -np.power((-np.sum(listRt_tec)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_tec))-1

    
### calculate annualized sharpe ratio
sr_den = (np.mean(listRt_den)-0.01) / np.std(listRt_den) * np.sqrt(252) 
sr_dwt = (np.mean(listRt_dwt)-0.01) / np.std(listRt_dwt) * np.sqrt(252)    
sr_pca = (np.mean(listRt_pca)-0.01) / np.std(listRt_pca) * np.sqrt(252) 
sr_tec = (np.mean(listRt_tec)-0.01) / np.std(listRt_tec) * np.sqrt(252)
#sr_bah = np.mean(ret[bah_id:bah_id+len(listRt_red)]) / np.std(ret[bah_id:bah_id+len(listRt_red)])* np.sqrt(252)    

### calculate mdd
den = np.cumsum(listRt_den)
trough_index = np.argmax(np.maximum.accumulate(den) - den)
peak_index = np.argmax(den[:trough_index])
mdd_den = (den[peak_index] - den[trough_index])/ den[peak_index]

dwt_ = np.cumsum(listRt_dwt)
trough_index = np.argmax(np.maximum.accumulate(dwt_) - dwt_)
peak_index = np.argmax(dwt_[:trough_index])
mdd_dwt = (dwt_[peak_index] - dwt_[trough_index])/ dwt_[peak_index]

pca_ = np.cumsum(listRt_pca)
trough_index = np.argmax(np.maximum.accumulate(pca_) - pca_)
peak_index = np.argmax(pca_[:trough_index])
mdd_pca = (pca_[peak_index] - pca_[trough_index])/ pca_[peak_index]

tec = np.cumsum(listRt_tec)
trough_index = np.argmax(np.maximum.accumulate(tec) - tec)
peak_index = np.argmax(tec[:trough_index])
mdd_tec = (tec[peak_index] - tec[trough_index])/ tec[peak_index]

# bah = np.cumsum(ret[bah_id:bah_id+len(listRt_red)])
# trough_index = np.argmax(np.maximum.accumulate(bah) - bah)
# peak_index = np.argmax(bah[:trough_index])
# mdd_bah = (bah[peak_index] - bah[trough_index])/ bah[peak_index]

### calculate calmar ratio
cr_den = ror_den / mdd_den
cr_dwt = ror_dwt / mdd_dwt
cr_pca = ror_pca / mdd_pca
cr_tec = ror_tec / mdd_tec
#cr_bah = ror_bah / mdd_bah

### calculate annualized sortino ratio 
# ddsig_den = np.std([i for i in listRt_den if i < 0.])

# ddsig_tec = np.std([i for i in listRt_tec if i < 0.])
# #ddsig_bah = np.std([i for i in ret[bah_id:bah_id+len(listRt_red)] if i < 0.])

# sro_den = np.mean(listRt_den) / ddsig_den * np.sqrt(252)

# sro_tec = np.mean(listRt_tec) / ddsig_tec * np.sqrt(252)
# #sro_bah = np.mean(ret[bah_id:bah_id+len(listRt_red)]) / ddsig_bah *np.sqrt(252)

print('np:', round(np_tec,2),'\t',round(np_pca,2),'\t',round(np_dwt,2),'\t',round(np_den,2),'\n', 
      'ror:', round(ror_tec,2),'\t', round(ror_pca,2),'\t', round(ror_dwt,2),'\t', round(ror_den,2), '\n',
      'sr:', round(sr_tec,2),'\t', round(sr_pca,2),'\t',round(sr_dwt,2),'\t', round(sr_den,2), '\n', 
      'mdd:',round(mdd_tec,2), '\t', round(mdd_pca,2), '\t',round(mdd_dwt,2), '\t',round(mdd_den,2), '\n',
      'cr:', round(cr_tec,2), '\t', round(cr_pca,2), '\t',round(cr_dwt,2), '\t',round(cr_den,2), '\n',)
      
### 'sro:', round(sro_tec,2), '\t', round(sro_den,2)

# print('PCA&DWT RRL Profit : {:.2f}'.format(np.sum(listRt_den)))
# print('Dwt RRL Profit : {:.2f}'.format(np.sum(listRt_dwt)))
# print('Pca RRL Profit : {:.2f}'.format(np.sum(listRt_pca)))
# print('Tec RRL Profit : {:.2f}'.format(np.sum(listRt_tec)))