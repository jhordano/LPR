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

data_r = pd.DataFrame((data.Close - data.Close.shift(1))/data.Close.shift(1))  
data_r['da'] = data.Date
data_r['dat'] = pd.to_datetime(data_r.da,format='%Y-%m-%d')
data_r['year'] = data_r.dat.dt.year
data_r['mm'] = data_r.dat.dt.month

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
        
    return np.sum(v) + 2*np.sum(c) 

data_s = pd.DataFrame(data_r.groupby(['year','mm'])['Close'].agg(var)).reset_index()


plt.plot(data_s.index ,data_s.Close)


