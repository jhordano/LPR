# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:55:09 2018

@author: HP PROBOOK
"""


import os
os.chdir('C:\\Users\\s3179575.WORKSPACE\\LPR')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

data = pd.read_table("sp500L.csv",sep=";")
data['Date'] = pd.to_datetime(data.Date,format='%Y%m%d')
data['year'] = data.Date.dt.year
#data['Return_excD'] = data['Return_excD']*100 
#data = data[data['year'] <= 1984] 

#data['Return'] = pd.DataFrame(((data.Close - data.Close.shift(1))/data.Close.shift(1))*100)  
#data['da'] = data.Date
#data['dat'] = pd.to_datetime(data.da,format='%Y-%m-%d')
#data['year'] = data.Date.dt.year
data['mm'] = data.Date.dt.month

#data_s = data_r.groupby(['year','mm'])['Close'].mean()        
#data_s = data_r.groupby(['year','mm'])['Close'].agg(lambda x: x.sum())        
def var(a):
    v = []
    c = []
    a = np.asarray(a)
    for i in range(0,len(a)-1):
        v.append(a[i]*a[i])
    for i in range(0,len(a)-2):    
        c.append(a[i+1]*a[i])
        
    return np.sqrt(np.sum(v) + 2*np.sum(c))

data_m = pd.DataFrame(data.groupby(['year','mm'])['Return_excD'].agg(var)).reset_index()
data_m = data_m.rename(columns={'Return_excD':'volatility'})
data_m['Date'] = pd.to_datetime(dict(year=data_m.year, month=data_m.mm,day=1))
data_m['Ln_vol'] = np.log(data_m['volatility'])
data_m['PC_vol'] = ((data_m.volatility - data_m.volatility.shift(1))/data_m.volatility.shift(1))*100  

data_m = data_m[data_m['year'] <= 1984] 
data_m = data_m[data_m['year'] >= 1928]

#data_s = data_s[data_s['year']>= 1953] 

mean = data_m.volatility.mean()
std = data_m.volatility.std()

mean_pc = data_m.PC_vol.mean()

print('Mean value',mean)
print('Standar deviation',std)

plt.plot(data_m.Date ,data_m.volatility)

model = ARIMA(data_m['Ln_vol'],order=(0,1,3))
model_fit = model.fit(disp=0)
print(model_fit.summary())

data_m['fit'] = model_fit.predict(typ='levels')
data_m['std_fit'] = np.exp(data_m['fit']  + 0.5*np.var(model_fit.resid))
data_m['var_fit'] = np.exp(2*data_m['fit']  + 2*np.var(model_fit.resid))

plt.plot(data_m.Date ,data_m.std_fit)

# %%


data = data[data['year'] >= 1953]
data_t = pd.read_table('Bonos.csv',sep=';')
data_t['Date'] = pd.to_datetime(data_t.Date,format='%d-%m-%Y')
data_t['year'] = data_t.Date.dt.year
data_t['month'] = data_t.Date.dt.month
data_t['select'] = data_t['year']*100 + data_t['month'] 

data_t = data_t[data_t['year'] <= 1984]

data_c = pd.merge(data,data_t,on='Date')


data_c['Date'].idxmax()
#df.loc[df['Value'].idxmax()]

#data_c.groupby(['select'])[['Date']].max().xs('DTB3',axis=1)

data_p =  data_c[data_c.groupby(['select'])['Date'].transform(max) == data_c['Date']]['DTB3']
data_p = data_p.rename(columns={'DTB3':'st'})
data_p = data_p.reindex(range(len(data_c)), method='bfill')

data_c['tbil'] = data_p 
# test_df2.reset_index(name='maxvalue').to_string(index=False)


# https://stackoverflow.com/questions/15705630/python-getting-the-row-which-has-the-max-value-in-groups-using-groupby


# %%

data_crsp = pd.read_table('crsp.csv',sep=',')
data_crsp['Date'] = pd.to_datetime(data_crsp.date,format='%Y%m')
data_crsp['crsp'] = data_crsp['Mkt-RF'] + data_crsp['RF']
data_crsp = data_crsp.rename(columns={'Mkt-RF':'spread'})
data_crsp['year'] = data_crsp.Date.dt.year
mean_crsp = np.mean(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1984)].crsp)

mean_crsp_1 = np.mean(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1984)].spread)
mean_crsp_2 = np.mean(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1952)].spread)
mean_crsp_3 = np.mean(data_crsp[(data_crsp['year']>=1953) & (data_crsp['year']<=1984)].spread)

std_crsp_1 = np.std(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1984)].spread)
std_crsp_2 = np.std(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1952)].spread)
std_crsp_3 = np.std(data_crsp[(data_crsp['year']>=1953) & (data_crsp['year']<=1984)].spread)


# http://pbpython.com/python-vis-flowchart.html


