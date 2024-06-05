# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:19:59 2021

@author: de
"""

### compare PCA+DWT RRL and DWT+PCA RRL


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
    
    X_pca = pca.fit_transform(X_ts)
    tmp = X_pca.copy() ## 
    X_pcadwt = discrete_wavelet_transform(tmp, 'haar', dec_level=4)
    

    test = StockTradingStrategy(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)    

#    listRt_pcadwt, listFt_pcadwt, listSr_pcadwt = test.walkForward(X_pcadwt, r_ts, p_ts, window_train, n_epochs, N) 

    listRt_pcadwt, listFt_pcadwt, listSr_pcadwt = test.walkForward(X_pcadwt, r_ts, window_train, n_epochs, N)                    

    return  listRt_pcadwt, listFt_pcadwt,  nanid  
 
def tradingDwtpca(market_info, 
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
    
    pca = PCA(n_components=0.95)
    
    X_dwtpca = pca.fit_transform(X_dwt)
    
    test = StockTradingStrategy(learning_rate =learning_rate,
                                lambda_re=lambda_re,
                                delta=delta,  
                                random_state=random_state)
          
#    listRt_dwtpca, listFt_dwtpca, listSr_dwtpca = test.walkForward(X_dwtpca, r_ts, p_ts, window_train, n_epochs, N)
    
    listRt_dwtpca, listFt_dwtpca, listSr_dwtpca = test.walkForward(X_dwtpca, r_ts, window_train, n_epochs, N)
      
    return   listRt_dwtpca, listFt_dwtpca, nanid #listRt_den, listFt_den, nanid 


# def plot_figure(x, y, z, tf):
    
#     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
    
#     ax[0].plot(x, y, 'k-v')
#     ax[0].set_xlabel('Transaction Cost Rates (%)')
#     ax[0].set_ylabel('Annualized Rate of Return (%)')
    
#     ax[1].plot(x, z, 'k-v')
#     ax[1].set_xlabel('Transaction Cost Rate (%)')
#     ax[1].set_ylabel('Annualized Sharpe Ratio')
    
#     ax[2].plot(x, tf, 'k-v')
#     ax[2].set_xlabel('Transaction Cost Rates (%)')
#     ax[2].set_ylabel('Trading Frequency (Times)')
    
#fname = 'data_old/szzs.csv'
fname = 'data_old/nyse.csv'
#fname = 'data_old/xom.csv'
#fname = 'data_old/crude-oil.csv'
#fname = 'data_old/zcf.csv'

param_pcadwt = {
'random_state': [42], #[np.random.randint(100000, size=1).item()], # 31561
'learning_rate': [0.1],
'lambda_re':  [0.01],
'delta':  [1],   #[0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01], 
'n_samples': [6000],
'window_train': [500],  
'n_epochs': [100],
'N':  [500]
}
  

# =============================================================================
# param_pca = {
# 'random_state': [42], #[np.random.randint(100000, size=1).item()], # 31561
# 'learning_rate': [0.1],
# 'lambda_re':  [0.01],
# 'delta': [0,1,2,3,4,5,6,7,8,9,10], #0.005 for sp500, artifi
# 'n_samples': [6000],
# 'window_train': [500],  
# 'n_epochs': [100],
# 'N':  [500]
# }
# =============================================================================

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

window_train = 500


#nyse_all = []
#xom_all = []
#zcf_all = []

for it in ParameterGrid(param_pcadwt):
    
    listRt_dwtpca, listFt_dwtpca, nanid  = tradingDwtpca(market_info, **it)
#    nyse_all.append(listRt_dwtpca)
#    xom_all.append(listRt_dwtpca)
#    zcf_all.append(listRt_dwtpca)
    
    listRt_pcadwt, listFt_pcadwt, nanid  = tradingPcadwt(market_info, **it)  
#    nyse_all.append(listRt_pcadwt)
#    xom_all.append(listRt_pcadwt)
#    zcf_all.append(listRt_pcadwt)
    
np_dwtpca = np.sum(listRt_dwtpca)
np_pcadwt = np.sum(listRt_pcadwt)
     
bah_id = nanid + window_train

if np.sum(listRt_dwtpca) >= 0:
    ror_dwtpca = np.power((np.sum(listRt_dwtpca)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_dwtpca))-1
else:
    ror_dwtpca = -np.power((-np.sum(listRt_dwtpca)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_dwtpca))-1    

if np.sum(listRt_pcadwt) >= 0:
    ror_pcadwt = np.power((np.sum(listRt_pcadwt)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_pcadwt))-1
else:
    ror_pcadwt = -np.power((-np.sum(listRt_pcadwt)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_pcadwt))-1
    
sr_dwtpca = (np.mean(listRt_dwtpca)-0.01) / np.std(listRt_dwtpca) * np.sqrt(252)  
sr_pcadwt = (np.mean(listRt_pcadwt)-0.01) / np.std(listRt_pcadwt) * np.sqrt(252) 

dwtpca = np.cumsum(listRt_dwtpca)
trough_index = np.argmax(np.maximum.accumulate(dwtpca) - dwtpca)
peak_index = np.argmax(dwtpca[:trough_index])
mdd_dwtpca = (dwtpca[peak_index] - dwtpca[trough_index])/ dwtpca[peak_index]

pcadwt = np.cumsum(listRt_pcadwt)
trough_index = np.argmax(np.maximum.accumulate(pcadwt) - pcadwt)
peak_index = np.argmax(pcadwt[:trough_index])
mdd_pcadwt = (pcadwt[peak_index] - pcadwt[trough_index])/ pcadwt[peak_index]

cr_dwtpca = ror_dwtpca / mdd_dwtpca
cr_pcadwt = ror_pcadwt / mdd_pcadwt

tf_dwtpca = [listFt_dwtpca[i-1]==listFt_dwtpca[i] for i in np.arange(1,len(listFt_dwtpca)) ]
tf_pcadwt = [listFt_pcadwt[i-1]==listFt_pcadwt[i] for i in np.arange(1,len(listFt_pcadwt)) ]
print(tf_dwtpca.count(0))
print(tf_pcadwt.count(0))

print('np:', round(np_dwtpca,2),'\t',round(np_pcadwt,2),'\n', 
      'ror:', round(ror_dwtpca,2),'\t', round(ror_pcadwt,2), '\n',
      'sr:', round(sr_dwtpca,2),'\t', round(sr_pcadwt,2), '\n', 
      'mdd:',round(mdd_dwtpca,2), '\t',round(mdd_pcadwt,2), '\n',
      'cr:', round(cr_dwtpca,2), '\t',round(cr_pcadwt,2), '\n',)

    
end = time.process_time()
print('running time:%s'%(end-start))

#np.save('D:/A_Mannheim/Quanttrade/results_new/nyse3_all.npy', np.array(nyse_all))
#np.save('D:/A_Mannheim/Quanttrade/results_new/xom3_all.npy', np.array(xom_all))
#np.save('D:/A_Mannheim/Quanttrade/results_new/zcf3_all.npy', np.array(zcf_all))
'''
def plot_all(nyse, xom, zcf):
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    ax[0].plot(np.cumsum(nyse[0]), label='DWT&PCA RRL')
    ax[0].plot(np.cumsum(nyse[1]), label='PCA&DWT RRL')
    ax[0].legend(loc = 'upper left')
    ax[0].set_xlabel('Trading Periods')
    ax[0].set_ylabel('Cumulative Profits ($)')
    
    ax[1].plot(np.cumsum(xom[0]), label='DWT&PCA RRL')
    ax[1].plot(np.cumsum(xom[1]), label='PCA&DWT RRL')
    ax[1].legend(loc = 'upper left')
    ax[1].set_xlabel('Trading Periods')
    ax[1].set_ylabel('Cumulative Profits ($)')
    
    ax[2].plot(np.cumsum(zcf[0]), label='DWT&PCA RRL')
    ax[2].plot(np.cumsum(zcf[1]), label='PCA&DWT RRL')
    ax[2].legend(loc = 'upper left')
    ax[2].set_xlabel('Trading Periods')
    ax[2].set_ylabel('Cumulative Profits ($)')
    
#plot(listRt_dwtpca, listRt_pcadwt)

#tmp = param_Pcadwt['delta']
#tmp = [i*100 for i in tmp]
#plot_figure(tmp, ror, sr, tf)
# np.save('D:/A_Mannheim/Quanttrade/data/t3zcf_tc.npy', np.array(den_all))
# np.save('D:/A_Mannheim/Quanttrade/data/t3zcf_tf.npy', np.array(tf))

nyse_all = np.load('D:/A_Mannheim/Quanttrade/results_new/nyse3_all.npy')
xom_all = np.load('D:/A_Mannheim/Quanttrade/results_new/xom3_all.npy')
zcf_all = np.load('D:/A_Mannheim/Quanttrade/results_new/zcf3_all.npy')
plot_all(nyse_all, xom_all, zcf_all)
'''

'''
pca_all = []
ror = []
sr = []
tf = []
for it in ParameterGrid(param_pca):
    listRt_den, listFt_den, nanid  = tradingPca(market_info, **it)  
    print(it['delta'], np.sum(listRt_den))
    pca_all.append(listRt_den)    
    bah_id = nanid + window_train
    if np.sum(listRt_den) >= 0:
       ror_den = np.power((np.sum(listRt_den)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_den))-1
    else:
       ror_den = -np.power((-np.sum(listRt_den)+market_info['Close'][bah_id])/market_info['Close'][bah_id], 252/len(listRt_den))-1    
    sr_den = np.mean(listRt_den) / np.std(listRt_den) * np.sqrt(252)  
    ror.append(ror_den * 100)
    sr.append(sr_den)
    tf_ = [listFt_den[i-1]==listFt_den[i] for i in np.arange(1,len(listFt_den)) ]
    tf.append(tf_.count(0))
end = time.process_time()
print('running time:%s'%(end-start))
tmp = param_pca['delta']
tmp = [i*100 for i in tmp]
plot_figure(tmp, ror, sr, tf)    
np.save('D:/A_Mannheim/Quanttrade/data/t3szzs_tc_pca.npy', np.array(pca_all))
np.save('D:/A_Mannheim/Quanttrade/data/t3szzs_tf_pca.npy', np.array(tf))
'''

'''
def plot_all(x, y1, y2, y3, y4, y5, z1, z2, z3, z4, z5, tf1, tf2, tf3, tf4):
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    
    ax[0].plot(x, y1, 'r-v', label='PCA&DWT RRL')
    ax[0].plot(x, y2, 'g-v', label='PCA RRL')
    ax[0].plot(x, y3, 'y-v', label='TA RRL')
    ax[0].plot(x, y4, 'b-v', label='Basic RRL')
    ax[0].plot(x, y5, 'k-v', label='B&H')
    ax[0].set_xlabel('Transaction Cost Rates ($)')
    ax[0].set_ylabel('Annualized Rate of Return (%)')
    ax[0].legend(loc='center right', bbox_to_anchor=(1, 0.5))
    
    ax[1].plot(x, z1, 'r-v', label='PCA&DWT RRL')
    ax[1].plot(x, z2, 'g-v', label='PCA RRL')
    ax[1].plot(x, z3, 'y-v', label='TA RRL')
    ax[1].plot(x, z4, 'b-v', label='Basic RRL')
    ax[1].plot(x, z5, 'k-v', label='B&H')
    ax[1].set_xlabel('Transaction Cost Rate ($)')
    ax[1].set_ylabel('Annualized Sharpe Ratio')
    ax[1].legend(loc='center right', bbox_to_anchor=(1,0.7))
    
    ax[2].plot(x, tf1, 'r-v', label='PCA&DWT RRL')
    ax[2].plot(x, tf2, 'g-v', label='PCA RRL')
    ax[2].plot(x, tf3, 'y-v', label='TA RRL')
    ax[2].plot(x, tf4, 'b-v', label='Basic RRL')
    ax[2].set_xlabel('Transaction Cost Rates ($)')
    ax[2].set_ylabel('Trading Frequency (Times)')
    ax[2].legend(loc='upper right', bbox_to_anchor=(1,1))
    
label = [0,1,2,3,4,5,6,7,8,9,10]        
nyse_pcadwt = np.load('D:/A_Mannheim/Quanttrade/data/t3nyse_tc_pcadwt.npy') 
nyse_pca =  np.load('D:/A_Mannheim/Quanttrade/data/t3nyse_tc_pca.npy') 
nyse_tec = np.load('D:/A_Mannheim/Quanttrade/data/t3nyse_tc_tec.npy')   
nyse_basic = np.load('D:/A_Mannheim/Quanttrade/data/t3nyse_tc_basic.npy')

ror1 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in nyse_pcadwt]
ror2 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in nyse_pca]
ror3 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in nyse_tec]
ror4 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in nyse_basic]
tmp_ = np.sum(ret[bah_id:bah_id+len(nyse_pcadwt[0])])
ror5 = [np.power((tmp_+tmp)/tmp, 252/len(nyse_pcadwt[0]))-1 if tmp_ >= 0 else -np.power((tmp_+tmp)/tmp, 252/len(nyse_pcadwt[0]))-1] * len(ror1)

sr1 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in nyse_pcadwt]
sr2 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in nyse_pca]
sr3 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in nyse_tec]
sr4 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in nyse_basic]
sr5 = [np.mean(ret[bah_id:bah_id+len(nyse_pcadwt[0])])/np.std(ret[bah_id:bah_id+len(nyse_pcadwt[0])])*np.sqrt(252)]*len(ror1)


tf1 = np.load('D:/A_Mannheim/Quanttrade/data/t3nyse_tf_pcadwt.npy') 
tf2 =  np.load('D:/A_Mannheim/Quanttrade/data/t3nyse_tf_pca.npy') 
tf3 = np.load('D:/A_Mannheim/Quanttrade/data/t3nyse_tf_tec.npy')   
tf4 = np.load('D:/A_Mannheim/Quanttrade/data/t3nyse_tf_basic.npy') 

plot_all(label,ror1,ror2,ror3,ror4,ror5,sr1,sr2,sr3,sr4,sr5,tf1,tf2,tf3,tf4)
'''
# =============================================================================
# label = [0,1,2,3,4,5,6,7,8,9,10]        
# s_pcadwt = np.load('D:/A_Mannheim/Quanttrade/data/t3szzs_tc_pcadwt.npy') 
# s_pca =  np.load('D:/A_Mannheim/Quanttrade/data/t3szzs_tc_pca.npy') 
# s_tec = np.load('D:/A_Mannheim/Quanttrade/data/t3szzs_tc_tec.npy')   
# s_basic = np.load('D:/A_Mannheim/Quanttrade/data/t3szzs_tc_basic.npy')
# 
# ror1 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in s_pcadwt]
# ror2 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in s_pca]
# ror3 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in s_tec]
# ror4 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in s_basic]
# tmp_ = np.sum(ret[bah_id:bah_id+len(s_pcadwt[0])])
# ror5 = [np.power((tmp_+tmp)/tmp, 252/len(s_pcadwt[0]))-1 if tmp_ >= 0 else -np.power((tmp_+tmp)/tmp, 252/len(s_pcadwt[0]))-1] * len(ror1)
# 
# sr1 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in s_pcadwt]
# sr2 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in s_pca]
# sr3 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in s_tec]
# sr4 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in s_basic]
# sr5 = [np.mean(ret[bah_id:bah_id+len(s_pcadwt[0])])/np.std(ret[bah_id:bah_id+len(s_pcadwt[0])])*np.sqrt(252)]*len(ror1)
# 
# 
# tf1 = np.load('D:/A_Mannheim/Quanttrade/data/t3szzs_tf_pcadwt.npy') 
# tf2 =  np.load('D:/A_Mannheim/Quanttrade/data/t3szzs_tf_pca.npy') 
# tf3 = np.load('D:/A_Mannheim/Quanttrade/data/t3szzs_tf_tec.npy')   
# tf4 = np.load('D:/A_Mannheim/Quanttrade/data/t3szzs_tf_basic.npy') 
# 
# plot_all(label,ror1,ror2,ror3,ror4,ror5,sr1,sr2,sr3,sr4,sr5,tf1,tf2,tf3,tf4)    
# =============================================================================

# =============================================================================
# label = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]        
# c_pcadwt = np.load('D:/A_Mannheim/Quanttrade/data/t3clf_tc_pcadwt.npy') 
# c_pca =  np.load('D:/A_Mannheim/Quanttrade/data/t3clf_tc_pca.npy') 
# c_tec = np.load('D:/A_Mannheim/Quanttrade/data/t3clf_tc_tec.npy')   
# c_basic = np.load('D:/A_Mannheim/Quanttrade/data/t3clf_tc_basic.npy')
# 
# ror1 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in c_pcadwt]
# ror2 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in c_pca]
# ror3 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in c_tec]
# ror4 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in c_basic]
# tmp_ = np.sum(ret[bah_id:bah_id+len(c_pcadwt[0])])
# ror5 = [np.power((tmp_+tmp)/tmp, 252/len(c_pcadwt[0]))-1 if tmp_ >= 0 else -np.power((tmp_+tmp)/tmp, 252/len(c_pcadwt[0]))-1] * len(ror1)
# 
# sr1 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in c_pcadwt]
# sr2 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in c_pca]
# sr3 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in c_tec]
# sr4 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in c_basic]
# sr5 = [np.mean(ret[bah_id:bah_id+len(c_pcadwt[0])])/np.std(ret[bah_id:bah_id+len(c_pcadwt[0])])*np.sqrt(252)]*len(ror1)
# 
# 
# tf1 = np.load('D:/A_Mannheim/Quanttrade/data/t3clf_tf_pcadwt.npy') 
# tf2 =  np.load('D:/A_Mannheim/Quanttrade/data/t3clf_tf_pca.npy') 
# tf3 = np.load('D:/A_Mannheim/Quanttrade/data/t3clf_tf_tec.npy')   
# tf4 = np.load('D:/A_Mannheim/Quanttrade/data/t3clf_tf_basic.npy') 
# 
# plot_all(label,ror1,ror2,ror3,ror4,ror5,sr1,sr2,sr3,sr4,sr5,tf1,tf2,tf3,tf4)     
# =============================================================================

# =============================================================================
# label = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]        
# z_pcadwt = np.load('D:/A_Mannheim/Quanttrade/data/t3zcf_tc_pcadwt.npy') 
# z_pca =  np.load('D:/A_Mannheim/Quanttrade/data/t3zcf_tc_pca.npy') 
# z_tec = np.load('D:/A_Mannheim/Quanttrade/data/t3zcf_tc_tec.npy')   
# z_basic = np.load('D:/A_Mannheim/Quanttrade/data/t3zcf_tc_basic.npy')
# 
# ror1 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in z_pcadwt]
# ror2 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in z_pca]
# ror3 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in z_tec]
# ror4 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in z_basic]
# tmp_ = np.sum(ret[bah_id:bah_id+len(z_pcadwt[0])])
# ror5 = [np.power((tmp_+tmp)/tmp, 252/len(z_pcadwt[0]))-1 if tmp_ >= 0 else -np.power((tmp_+tmp)/tmp, 252/len(z_pcadwt[0]))-1] * len(ror1)
# 
# sr1 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in z_pcadwt]
# sr2 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in z_pca]
# sr3 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in z_tec]
# sr4 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in z_basic]
# sr5 = [np.mean(ret[bah_id:bah_id+len(z_pcadwt[0])])/np.std(ret[bah_id:bah_id+len(z_pcadwt[0])])*np.sqrt(252)]*len(ror1)
# 
# 
# tf1 = np.load('D:/A_Mannheim/Quanttrade/data/t3zcf_tf_pcadwt.npy') 
# tf2 =  np.load('D:/A_Mannheim/Quanttrade/data/t3zcf_tf_pca.npy') 
# tf3 = np.load('D:/A_Mannheim/Quanttrade/data/t3zcf_tf_tec.npy')   
# tf4 = np.load('D:/A_Mannheim/Quanttrade/data/t3zcf_tf_basic.npy') 
# 
# plot_all(label,ror1,ror2,ror3,ror4,ror5,sr1,sr2,sr3,sr4,sr5,tf1,tf2,tf3,tf4)     
# =============================================================================
    
# =============================================================================
# label = [0,1,2,3,4,5,6,7,8,9,10]        
# x_pcadwt = np.load('D:/A_Mannheim/Quanttrade/data/t3xom_tc_pcadwt.npy') 
# x_pca =  np.load('D:/A_Mannheim/Quanttrade/data/t3xom_tc_pca.npy') 
# x_tec = np.load('D:/A_Mannheim/Quanttrade/data/t3xom_tc_tec.npy')   
# x_basic = np.load('D:/A_Mannheim/Quanttrade/data/t3xom_tc_basic.npy')
# 
# ror1 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in x_pcadwt]
# ror2 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in x_pca]
# ror3 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in x_tec]
# ror4 = [np.power((np.sum(i)+tmp)/tmp, 252/len(i))-1 if np.sum(i)>=0 else -np.power((-np.sum(i)+tmp)/tmp, 252/len(i))-1 for i in x_basic]
# tmp_ = np.sum(ret[bah_id:bah_id+len(x_pcadwt[0])])
# ror5 = [np.power((tmp_+tmp)/tmp, 252/len(x_pcadwt[0]))-1 if tmp_ >= 0 else -np.power((tmp_+tmp)/tmp, 252/len(x_pcadwt[0]))-1] * len(ror1)
# 
# sr1 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in x_pcadwt]
# sr2 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in x_pca]
# sr3 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in x_tec]
# sr4 = [np.mean(i)/np.std(i) * np.sqrt(252) for i in x_basic]
# sr5 = [np.mean(ret[bah_id:bah_id+len(x_pcadwt[0])])/np.std(ret[bah_id:bah_id+len(x_pcadwt[0])])*np.sqrt(252)]*len(ror1)
# 
# 
# tf1 = np.load('D:/A_Mannheim/Quanttrade/data/t3xom_tf_pcadwt.npy') 
# tf2 =  np.load('D:/A_Mannheim/Quanttrade/data/t3xom_tf_pca.npy') 
# tf3 = np.load('D:/A_Mannheim/Quanttrade/data/t3xom_tf_tec.npy')   
# tf4 = np.load('D:/A_Mannheim/Quanttrade/data/t3xom_tf_basic.npy') 
# 
# plot_all(label,ror1,ror2,ror3,ror4,ror5,sr1,sr2,sr3,sr4,sr5,tf1,tf2,tf3,tf4)     
# =============================================================================

