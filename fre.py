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
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

data = pd.read_table("sp500L.csv",sep=";")
data['Date'] = pd.to_datetime(data.Date,format='%Y%m%d')
data['year'] = data.Date.dt.year
data['Return_excD'] = data['Return_excD']

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
data_m['volatility_2'] = np.power(data_m['volatility'],2)

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
plt.plot(data_m.Date ,data_m.std_fit)

data_m = data_m.reset_index(drop=True)

data_m.set_value(0,'std_fit' , data_m.iloc[0]['volatility'])
data_m['Unp_std'] = data_m['volatility'] - data_m['std_fit']

data_m['var_fit'] = np.exp(2*data_m['fit']  + 2*np.var(model_fit.resid))
data_m.set_value(0,'var_fit' , data_m.iloc[0]['volatility_2'])
data_m['Unp_var'] = data_m['volatility_2'] - data_m['var_fit']






# %% ARCH model

#data = data[data['year'] >= 1953]

data_t = pd.read_table('Bonos.csv',sep=';')
data_t['Date'] = pd.to_datetime(data_t.Date,format='%d-%m-%Y')
data_t['year'] = data_t.Date.dt.year
data_t['month'] = data_t.Date.dt.month
data_t['select'] = data_t['year']*100 + data_t['month'] 

data_t = data_t[data_t['year'] <= 1984]
data_c = pd.merge(data,data_t,on='Date')

#data_c['Date'].idxmax()
#df.loc[df['Value'].idxmax()]

#data_c.groupby(['select'])[['Date']].max().xs('DTB3',axis=1)
data_bo_m =  data_c[data_c.groupby(['select'])['Date'].transform(max) == data_c['Date']]['DTB3']
data_bo_m = data_bo_m.rename(columns={'DTB3':'st'})
data_bo_m = data_bo_m.reindex(range(len(data_c)), method='bfill')

data_c['tbil'] = data_bo_m/12 
# test_df2.reset_index(name='maxvalue').to_string(index=False)


# https://stackoverflow.com/questions/15705630/python-getting-the-row-which-has-the-max-value-in-groups-using-groupby


# %%

data_crsp = pd.read_table('crsp.csv',sep=',')
data_crsp['Date'] = pd.to_datetime(data_crsp.date,format='%Y%m')
data_crsp['crsp'] = data_crsp['Mkt-RF'] + data_crsp['RF']
data_crsp = data_crsp.rename(columns={'Mkt-RF':'spread'})
data_crsp['year'] = data_crsp.Date.dt.year

data_RF = pd.DataFrame(data=np.array(data_crsp['RF']), index=data_crsp['Date'])
data_RF_d = data_RF.reindex(data['Date'],method='bfill')

#mean_crsp = np.mean(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1984)].crsp)
mean_crsp_1 = np.mean(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1984)].spread)
mean_crsp_2 = np.mean(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1952)].spread)
mean_crsp_3 = np.mean(data_crsp[(data_crsp['year']>=1953) & (data_crsp['year']<=1984)].spread)

std_crsp_1 = np.std(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1984)].spread)
std_crsp_2 = np.std(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1952)].spread)
std_crsp_3 = np.std(data_crsp[(data_crsp['year']>=1953) & (data_crsp['year']<=1984)].spread)

# http://pbpython.com/python-vis-flowchart.html

plt.plot(data_RF_d.index,data_RF_d[0])
plt.plot(data_c.Date,data_c.tbil)

data['RF'] = np.array(data_RF_d[0])
data['spread'] = data['Return_excD'] - data['RF'] 


# %%
data = data[data['year'] <= 1984]
data = data[data['year'] >= 1928]

#from arch import arch_model
#from arch.univariate import ConstantMean, GARCH, Normal
from arch.univariate import ARX, GARCH, Normal

am = ARX(data['spread'],lags=1)
am.volatility = GARCH(2, 0, 1)
am.distribution = Normal()

res = am.fit()

res.summary()

# %%
import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
import math 

# (1/np.sqrt(scale*2*math.pi))*np.exp(-0.5*np.power((1-loc)/scale,2))
    
# Estimation of a GARCH model by ML.

def cun_pmf(x, mu, theta, a, b, c_1 , c_2):
    # compute residuals
    n = len(x)
    e = np.zeros(n)    
    ss = np.sum(e*e)/(n-4)
    # generate sigt and log likelihood
    sigt = np.zeros(n);
    loglik = np.zeros(n);

    for i in range(n):
        if i <= 1:
            sigt[i] = a + b*ss + c_1*ss + c_2*ss;
            e[i] = x[i] - mu  
        else:
            sigt[i] = a + b*sigt[i-1] + c_1*e[i-1]*e[i-1] + c_2*e[i-2]*e[i-2];
            e[i] = x[i] - mu -theta*e[i-1]
            
        loglik[i] = -0.5*(np.log(2*math.pi) + np.log(sigt[i]) + (e[i]*e[i])/sigt[i]);
                
    return -np.sum(loglik)

# %%
class arch_c(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(arch_c, self).__init__(endog, exog, **kwds)
     
    
    def nloglikeobs(self, params):
        mu = params[0]
        theta = params[1]
        a = params[2]
        b = params[3]
        c_1 = params[4]
        c_2 = params[5]

        return cun_pmf(self.endog,mu,theta,a,b,c_1,c_2)
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append('theta')
        self.exog_names.append('a')
        self.exog_names.append('b')
        self.exog_names.append('c_1')
        self.exog_names.append('c_2')        
        start_params = np.array([0.1,0.1, 0.5,1,0.5,0.1])
            
        return super(arch_c, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)

model = arch_c(data['spread'])
results = model.fit()
results.summary()

# %%
data_m = data_m.reset_index(drop=True)
data_m['spread'] = np.array(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1984)].spread/100)

#data_m = data_m[(data_m['year']>=1953) & (data_m['year']<=1984)]
mean_crsp = np.mean(data_m['spread'])
std_crsp = np.std(data_m['spread'])
    
mean_crsp_b = np.sum(data_m['spread']/data_m['volatility'])/(np.sum(1/data_m['volatility']))
mean_crsp_c = np.sum(data_m['spread']/data_m['std_fit'])/(np.sum(1/data_m['std_fit']))
# std_fit

res_ols = sm.OLS(data_m['spread'], np.zeros(len(data_m['spread']))-1).fit()
print(res_ols.summary())

mod_wls = sm.WLS(data_m['spread'], np.zeros(len(data_m['spread']))-1, weights=1./data_m['volatility'])
res_wls = mod_wls.fit()
print(res_wls.summary())


# %% Part 3     Estimating relations between risk premiums and volatility

#mod_wls = sm.WLS(data_m['spread'], data_m['std_fit'], weights=1./data_m['volatility'])
#res_wls = mod_wls.fit()
#print(res_wls.summary())

#X = np.concatenate((np.array(data_m['std_fit']).reshape(684,1) , np.ones( (len(data_m['std_fit']),1))), axis=1)
X = np.concatenate(( np.ones( (len(data_m['std_fit']),1)) , np.array(data_m['std_fit']).reshape(684,1) ) , axis=1)
y = np.array(data_m['spread'])
Om = np.diag(1/data_m['std_fit'])

beta = np.matmul( np.linalg.inv(np.matmul(np.matmul(np.transpose(X),Om),X)), np.matmul(np.matmul(np.transpose(X),Om),y) )
      
X_1 = np.concatenate(( np.ones( (len(data_m['std_fit']),1)) , np.array(data_m['std_fit']).reshape(684,1) , np.array(data_m['Unp_std']).reshape(684,1) ) , axis=1)
beta_1 = np.matmul( np.linalg.inv(np.matmul(np.matmul(np.transpose(X_1),Om),X_1)), np.matmul(np.matmul(np.transpose(X_1),Om),y) )



X_v = np.concatenate(( np.ones( (len(data_m['var_fit']),1)) , np.array(data_m['var_fit']).reshape(684,1) ) , axis=1)

beta_v = np.matmul( np.linalg.inv(np.matmul(np.matmul(np.transpose(X),Om),X)), np.matmul(np.matmul(np.transpose(X),Om),y) )     
X_1 = np.concatenate(( np.ones( (len(data_m['var_fit']),1)) , np.array(data_m['var_fit']).reshape(684,1) , np.array(data_m['Unp_var']).reshape(684,1) ) , axis=1)
beta_v_1 = np.matmul( np.linalg.inv(np.matmul(np.matmul(np.transpose(X_1),Om),X_1)), np.matmul(np.matmul(np.transpose(X_1),Om),y) )
