# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:55:09 2018

@author: HP PROBOOK
"""


import os
os.chdir('C:\\Users\\s3179575\\LPR')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_table("sp500.csv",sep=",")
data['Date'] = pd.to_datetime(data.Date,format='%Y-%m-%d')
data['year'] = data.Date.dt.year
data = data[data['year'] <= 1984] 

data['Return'] = pd.DataFrame(((data.Close - data.Close.shift(1))/data.Close.shift(1))*100)  
data['da'] = data.Date
data['dat'] = pd.to_datetime(data.da,format='%Y-%m-%d')
data['year'] = data.dat.dt.year
data['mm'] = data.dat.dt.month

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

data_s = pd.DataFrame(data.groupby(['year','mm'])['Return'].agg(var)).reset_index()
data_s = data_s[data_s['year']>= 1953] 

mean = data_s.Return.mean()
std = data_s.Return.std()

print('Mean value',mean)
print('Standar deviation',std)

plt.plot(data_s.index ,data_s.Return)

# %%
data = data[data['year'] >= 1953]
data_t = pd.read_table('Bonos.csv',sep=';')
data_t['Date'] = pd.to_datetime(data_t.Date,format='%d-%m-%Y')
data_t['year'] = data_t.Date.dt.year
data_t = data_t[data_t['year'] <= 1984]
data_c = pd.merge(data,data_t,on='Date')
data_c['average'] = data_c.groupby(['year_y'])['DTB3'].transform(max)


# test_df2.reset_index(name='maxvalue').to_string(index=False)


# https://stackoverflow.com/questions/15705630/python-getting-the-row-which-has-the-max-value-in-groups-using-groupby
