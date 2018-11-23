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
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from tabulate import tabulate
from scipy.stats import skew
from statsmodels.base.model import GenericLikelihoodModel
import math 
from scipy import stats

# ** Daily data will be stored in the data frame data. This cotains the S&P returns, with the label Return_excD. 
#    
# ** Mointhly data will be stores in the data frame data_m. This contains the CRSP returns

data = pd.read_table("sp500L.csv",sep=";")
data['Date'] = pd.to_datetime(data.Date,format='%Y%m%d')
data['year'] = data.Date.dt.year
data['Return_excD'] = data['Return_excD']
data['mm'] = data.Date.dt.month
# Calculations of daily volatility
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

data_m['Date'] = pd.to_datetime(dict(year=data_m.year, month=data_m.mm, day=1))
data_m['Ln_vol'] = np.log(data_m['volatility'])


# %%
#   Estimation of the predicted volatility
model = ARIMA(data_m[(data_m['Date']>'1927-11-01 00:00:00') & (data_m['year'] <= 1984)]['Ln_vol'],order=(0,1,3))
model_fit = model.fit(disp=0)
data_m['fit_0'] = model_fit.predict(typ='levels')
data_m['std_fit_0'] = np.exp(data_m['fit_0']  + 0.5*np.var(model_fit.resid))
data_m['Unp_std_0'] = data_m['volatility'] - data_m['std_fit_0']
data_m['var_fit_0'] = np.exp(2*data_m['fit_0']  + 2*np.var(model_fit.resid))
data_m['Unp_var_0'] = data_m['volatility_2'] - data_m['var_fit_0']

model_1 = ARIMA(data_m[(data_m['Date']>'1927-11-01 00:00:00')]['Ln_vol'],order=(0,1,3))
model_fit_1 = model_1.fit(disp=0)
data_m['fit_1'] = model_fit_1.predict(typ='levels')
data_m['std_fit_1'] = np.exp(data_m['fit_1']  + 0.5*np.var(model_fit_1.resid))
data_m['Unp_std_1'] = data_m['volatility'] - data_m['std_fit_1']
data_m['var_fit_1'] = np.exp(2*data_m['fit_1']  + 2*np.var(model_fit_1.resid))
data_m['Unp_var_1'] = data_m['volatility_2'] - data_m['var_fit_1']


plt.plot(data_m.Date, data_m.volatility)
plt.plot(data_m.Date ,data_m.std_fit_0)

# %%
class table_1(object):
    def __init__(self, data, year_comp):
        #self.f = f
        self.data = data
        self.year_comp = year_comp        
    def compt(self):
        model = ARIMA(self.data['Ln_vol'] ,order=(0,1,3))
        model_fit = model.fit(start_params = np.array([0, 0, 0, 0]) ,disp=0)
        return model_fit
   
    def table_comp_a(self):
        year_comp = self.year_comp        
        mean = self.data.volatility.mean()
        std = self.data.volatility.std()
        skw = skew(self.data.volatility)
        ind = 2
        if year_comp == 0:
            fre_p = []
            ind = 1
        elif year_comp == 1:
            fre_p = np.array([0.0474, 0.0325, 2.80])    
        elif year_comp == 2:
            fre_p = np.array([0.0607, 0.0417, 2.08])    
        else:
            fre_p = np.array([0.0371, 0.0168, 1.70])    
        H1 = np.array(["mean", "std dev", "Skewness"])        
        coef = np.array([mean,std,skw])        
        d_p = np.concatenate((fre_p,coef),axis=0).reshape(ind,-1)
        table = tabulate(d_p , headers = H1, floatfmt=".4f") 
        return table          

    def table_comp_b(self):
        t = self.compt()
        year_comp = self.year_comp
        ind = 2
        if year_comp == 0:
            fre_p = []
            ind = 1
        elif year_comp == 1:
            fre_p = np.array([0, 0.524, 0.158, 0.09])    
        elif year_comp == 2:
            fre_p = np.array([-0.0012, 0.552, 0.193, 0.031])    
        else:
            fre_p = np.array([0.0010, 0.506, 0.097 , 0.161])                 
        H1 = np.array(["theta_0", "theta_1", "theta_2", "theta_3"])        
        coef = np.array(t.params)        
        d_p = np.concatenate((fre_p,coef),axis=0).reshape(ind,-1)
        table = tabulate(d_p , headers = H1, floatfmt=".4f") 
        return table              
    
tab_1 = table_1(data_m[(data_m['year'] >= 1928) & (data_m['year'] <= 1984)] , 1)
print(tab_1.table_comp_a())


# %% Treasury Bill data
data_t = pd.read_table('Bonos.csv',sep=';')
data_t['Date'] = pd.to_datetime(data_t.Date,format='%d-%m-%Y')
data_t['yy'] = data_t.Date.dt.year
data_t['month'] = data_t.Date.dt.month
data_t['select'] = data_t['yy']*100 + data_t['month'] 

data_c = pd.merge(data,data_t,on='Date')

# Selecting the return of the last day of the month
data_bo_m_1 =  data_c[data_c.groupby(['select'])['Date'].transform(max) == data_c['Date']]
data_bo_m_1['Ind'] = data_bo_m_1.index 
data_bo_m_1['w_day'] = data_bo_m_1['Ind'] - data_bo_m_1['Ind'].shift(1)
data_bo_m_1.loc[19,'w_day'] = 19
# Selecting the return of the last day of the month
data_bo_m =  pd.DataFrame(data_c[data_c.groupby(['select'])['Date'].transform(max) == data_c['Date']]['DTB3'])
data_bo_m = data_bo_m.rename(columns={'DTB3':'TB'})
data_bo_m['TB'] =  data_bo_m['TB']/data_bo_m_1['w_day']

data_bo_m = data_bo_m.reindex(range(len(data_c)), method='bfill')
data_c['tbil'] = data_bo_m
#plt.plot(data_c.Date, data_c.tbil)
#plt.plot(data_c.Date ,data_c.DTB3)

# %%
data_crsp = pd.read_table('crsp.csv',sep=',')
data_crsp['Date'] = pd.to_datetime(data_crsp.date,format='%Y%m')
data_crsp['crsp'] = data_crsp['Mkt-RF'] + data_crsp['RF']
data_crsp = data_crsp.rename(columns={'Mkt-RF':'spread'})
data_crsp['yy'] = data_crsp.Date.dt.year
    
data_RF = pd.DataFrame(data=np.array(data_crsp['RF']), index=data_crsp['Date'])
data_RF_d = data_RF.reindex(data['Date'],method='bfill')

#mean_crsp_1 = np.mean(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1984)].spread)
#std_crsp_1 = np.std(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1984)].spread)

plt.plot(data_RF_d.index,data_RF_d[0]/2)
plt.plot(data_c.Date,data_c.tbil )

data['RF'] = np.array(data_RF_d[0]/2)
data['spread'] = data['Return_excD'] - data['RF']/100 

# %%
#from arch import arch_model
#from arch.univariate import ConstantMean, GARCH, Normal
#from arch.univariate import ARX, GARCH, Normal
#am = ARX(data['spread'],lags=1)
#am.volatility = GARCH(2, 0, 1)
#am.distribution = Normal()
#res = am.fit()
#res.summary()
# %% # %% ARCH model (Estimation of a GARCH model by ML)

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
            e[i] = x[i] - mu + theta*e[i-1]
            
        loglik[i] = -0.5*(np.log(2*math.pi) + np.log(sigt[i]) + (e[i]*e[i])/sigt[i]);
                
    return -np.sum(loglik)

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

# %%
        

    
class table_2(object):
    def __init__(self, data, year_comp):
        self.data = data
        self.year_comp = year_comp        
    def compt(self):
        model = arch_c(self.data['spread']*100)
        model_fit = model.fit()
        return model_fit   
    def table(self):
        t = self.compt()
        coef = np.array(t.params)
        H1 = np.array(["alpha", "theta", "a", "b", "c_1" , "c_2"])
        table = tabulate([coef], headers = H1, floatfmt=".4f") 
        return table    
    
    def table_comp_a(self):
        t = self.compt()
        year_comp = self.year_comp
        ind = 2
        if year_comp == 0:
            fre_p = []
            ind = 1
        elif year_comp == 1:
            fre_p = np.array([0.000324, -0.157, 0.00000062, 0.919, 0.121, -0.044])    
        elif year_comp == 2:
            fre_p = np.array([0.000496, -0.090, 0.00000149, 0.898, 0.106, -0.012])    
        else:
            fre_p = np.array([0.000257, -0.211, 0.00000052, 0.922, 0.130, -0.060])    
        H1 = np.array(["alpha", "theta", "a", "b", "c_1" , "c_2"])
        coef = np.array(t.params)        
        d_p = np.concatenate((fre_p,coef),axis=0).reshape(ind,-1)
        table = tabulate(d_p , headers = H1, floatfmt=".4f") 
        return table              
    
tab_2 = table_2(data[(data['year'] >= 1953) & (data['year'] <= 1984)] , 3)
print(tab_2.table_comp_a())




# %%

data_m_c = pd.merge(data_m,data_crsp,on='Date')
data_m_c['spread'] = data_m_c['spread']/100
#data_m = data_m.reset_index(drop=True)
#data_m['spread'] = np.array(data_crsp[(data_crsp['year']>=1928) & (data_crsp['year']<=1984)].spread/100)

mean_crsp = np.mean(data_m_c['spread'])
std_crsp = np.std(data_m_c['spread'])
    
#mean_crsp_b = np.sum(data_m_c['spread']/data_m_c['volatility'])/(np.sum(1/data_m_c['volatility']))
#mean_crsp_c = np.sum(data_m_c['spread']/data_m_c['std_fit'])/(np.sum(1/data_m_c['std_fit']))

# std_fit
#res_ols = sm.OLS(data_m_c['spread'], np.zeros(len(data_m_c['spread']))).fit()
#print(res_ols.summary())

#mod_wls = sm.WLS(data_m_c['spread'], np.ones(len(data_m_c['spread'])), weights=1./data_m_c['volatility'])
#res_wls = mod_wls.fit()
#print(res_wls.summary())

# %%

class table_3(object):
    def __init__(self, data, year_comp):
        #self.f = f
        self.data = data
        self.year_comp = year_comp      
               
    def table_comp_a(self):
        year_comp = self.year_comp
        ind = 2
        std_est = 'std_fit_0'
        
        if year_comp == 0:
            fre_p = []
            ind = 1
            A,B = 1928,2017
            std_est = 'std_fit_1'
            
        elif year_comp == 1:
            fre_p = np.array([0.0061, 0.0116, 0.0055, 0.0579])
            A,B = 1928,1984
        elif year_comp == 2:
            fre_p = np.array([0.0074, 0.0151, 0.0083, 0.0742])
            A,B = 1928, 1952
        else:
            fre_p = np.array([0.0050, 0.0102 ,0.0044 ,0.0410]) 
            A,B = 1953,1984

        self.data = self.data[(self.data['year'] >= A) & (self.data['year'] <= B)]            
        mean = self.data.spread.mean()
        std = self.data.spread.std()
        WLS_b = np.sum(self.data['spread']/self.data['volatility'])/(np.sum(1/self.data['volatility']))
        WLS_c = np.sum(self.data['spread']/self.data[std_est])/(np.sum(1/self.data[std_est]))        
        H1 = np.array(["Mean", "WLS b", "WLS c", "std dev"])        
        coef = np.array([mean, WLS_b , WLS_c ,std])        
        d_p = np.concatenate((fre_p,coef),axis=0).reshape(ind,-1)
        table = tabulate(d_p , headers = H1, floatfmt=".4f") 
        return table          

tab_3 = table_3(data_m_c, 1)
tab_3_1 = tab_3.table_comp_a()


# %% Part 3     Estimating relations between risk premiums and volatility
data_m_c_a = data_m_c[(data_m_c['year'] >= 1928) & (data_m_c['year'] <= 1984)]

X = np.concatenate(( np.ones( (len(data_m_c_a['var_fit_0']),1)) , np.array(data_m_c_a['std_fit_0']).reshape(-1,1) ) , axis=1)
y = np.array(data_m_c_a['spread'])

model_s = sm.WLS(y, X, weights = 1/data_m_c_a['volatility'])
results_s = model_s.fit()
print(results_s.summary())

X_1 = np.concatenate(( np.ones( (len(data_m_c_a['std_fit_0']),1)) , np.array(data_m_c_a['std_fit_0']).reshape(-1,1) , np.array(data_m_c_a['Unp_var_0']).reshape(-1,1) ) , axis=1)
model_s_u = sm.WLS(y,X_1, weights = 1/data_m_c_a['volatility'])
results_s_u = model_s_u.fit()
print(results_s_u.summary())

X_v = np.concatenate(( np.ones( (len(data_m_c_a['var_fit_0']),1)) , np.array(data_m_c_a['var_fit_0']).reshape(684,1) ) , axis=1)
model_v = sm.WLS(y, X_v, weights = 1/data_m_c_a['std_fit_0'])
results_v = model_v.fit()
print(results_v.summary())
     
X_1 = np.concatenate(( np.ones( (len(data_m_c_a['var_fit_0']),1)) , np.array(data_m_c_a['var_fit_0']).reshape(684,1) , np.array(data_m_c_a['Unp_var']).reshape(684,1) ) , axis=1)
model_v_u = sm.WLS(y, X_1, weights = 1/data_m_c_a['std_fit_0'])
results_v_u = model_v_u.fit()
print(results_v_u.summary())


# %%

class table_4(object):
    def __init__(self, data, year_comp):
        self.data = data
        self.year_comp = year_comp      
        
    def predic(self, S, E, est):
        # standard deviation
        self.data = self.data[(self.data['year'] >= S) & (self.data['year'] <= E)]
        X = np.concatenate(( np.ones( (len(self.data['std_fit_'+ est]),1)) , np.array(self.data['std_fit_'+ est]).reshape(-1,1) ) , axis=1)
        y = np.array(self.data['spread'])        
        model_s = sm.WLS(y, X, weights = 1/self.data['volatility'])
        results_s = model_s.fit()

        X_1 = np.concatenate(( np.ones( (len(self.data['std_fit_'+ est]),1)) , np.array(self.data['std_fit_'+ est]).reshape(-1,1) , np.array(self.data['Unp_std_'+est]).reshape(-1,1) ) , axis=1)
        model_s_u = sm.WLS(y,X_1, weights = 1/self.data['volatility'])
        results_s_u = model_s_u.fit()

        # Variance deviation
        self.data = self.data[(self.data['year'] >= S) & (self.data['year'] <= E)]
        X_v = np.concatenate(( np.ones( (len(self.data['var_fit_'+ est]),1)) , np.array(self.data['var_fit_'+ est]).reshape(-1,1) ) , axis=1)
        model_s_v = sm.WLS(y, X_v, weights = 1/self.data['volatility'])
        results_s_v = model_s_v.fit()

        X_1_v = np.concatenate(( np.ones( (len(self.data['var_fit_'+ est]),1)) , np.array(self.data['var_fit_'+ est]).reshape(-1,1) , np.array(self.data['Unp_var_'+est]).reshape(-1,1) ) , axis=1)
        model_s_u_v = sm.WLS(y,X_1_v, weights = 1/self.data['volatility'])
        results_s_u_v = model_s_u_v.fit()        
        return np.concatenate((results_s.params , results_s_u.params , results_s_v.params , results_s_u_v.params)).reshape(2,5)
               
    def table_comp_a(self):
        year_comp = self.year_comp
        ind = 4
        est = '0'
        
        if year_comp == 0:
            fre_p = []
            ind = 2
            S,E = 1928,2017
            est = '1'
        elif year_comp == 1:
            fre_p = np.array([[0.0047, 0.023, 0.0077 , -0.050 ,-1.010 ],[0.0050, 0.335, 0.0057 , 0.088 , -4.438 ]])
            S,E = 1928,1984
        elif year_comp == 2:
            fre_p = np.array([[0.0142, -0.133 , 0.0199 , -0.230 , -1.007],[0.0092, -0.324 , 0.0144 , -0.671 , -3.985]])
            S,E = 1928, 1952
        else:
            fre_p = np.array([[0.0027, 0.055 ,0.0068 , -0.071, -1.045],[0.0031, 1.058 , 0.0046 , -0.349, -9.075]]) 
            S,E = 1953,1984
            
        coef = self.predic(S,E, est)                    
        H1 = np.array(["alpha", "Beta", "alpha", "beta" , "gamma"])        
        d_p = np.concatenate((fre_p[0],coef[0], fre_p[1], coef[1]),axis=0).reshape(ind,-1)
        table = tabulate(d_p , headers = H1, floatfmt=".4f") 
        return table          

tab_4 = table_4(data_m_c, 3)
tab_4_1 = tab_4.table_comp_a()
print(tab_4_1)






# %% Estimation of the GARCH-M model
from arch.univariate import ConstantMean, GARCH

def lik_g_m(x, mu,beta,theta,a,b,c_1,c_2):
    
    # compute residuals
    n = len(x)
    ss = np.std(x)
    e = np.zeros(n)    
    # generate sigt and log likelihood
    sigt = np.zeros(n)  ;
    loglik = np.zeros(n);

    for i in range(n):
        if i <= 1:
            sigt[i] = a + b*ss + c_1 + c_2;
            e[i] = x[i] - mu - theta*ss 
            
            loglik[i] = -0.5*( np.log(2*math.pi) + np.log(sigt[i]) + (e[i]*e[i])/sigt[i] )            
        else:
            sigt[i] = a + b*sigt[i-1] + c_1*e[i-1]*e[i-1] + c_2*e[i-2]*e[i-2] 
            e[i] = x[i] - mu - beta*np.sqrt(sigt[i]) + theta*e[i-1]
#            e[i] = x[i] - mu - beta*sigt[i] + theta*e[i-1]
            loglik[i] = -0.5*( np.log(2*math.pi) + np.log(sigt[i]) + (e[i]*e[i])/sigt[i] )
                
    return -np.sum(loglik)

class garch_m(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(garch_m, self).__init__(endog, exog, **kwds)
     
    
    def nloglikeobs(self, params):
        mu = params[0]
        beta = params[1]
        theta = params[2]
        a = params[3]
        b = params[4]
        c_1 = params[5]
        c_2 = params[6]

        return lik_g_m(self.endog,mu,beta,theta,a,b,c_1,c_2)
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append('beta')
        self.exog_names.append('theta')
        self.exog_names.append('a')
        self.exog_names.append('b')
        self.exog_names.append('c_1')
        self.exog_names.append('c_2')
        
        gar_0 = ConstantMean(data['spread'])
        gar_0.volatility = GARCH(p=2, q=1)
        gar_0_r = gar_0.fit()
        gar_pa_0 = np.array(gar_0_r.params)
        sigma_2 = gar_0_r.conditional_volatility
#        sigma_2 = np.sqrt(gar_0_r.conditional_volatility)
        
        mean_0 = statsmodels.tsa.arima_model.ARMA( data['spread'] ,exog=sigma_2,order=(0,1))
        mean_0_r = mean_0.fit()
        mean_pa_0 = np.array(mean_0_r.params)        
        
  #       start_params = np.concatenate([ [-0.001],[0.073],[-0.157] , [gar_pa_0[1]] , [gar_pa_0[4]] , gar_pa_0[2:4]])        
  #   start_params = np.array([ -0.001, 0.073, -0.157 , 0.00006 , 0.918 , 0.121, -0.043 ])        
        start_params = np.concatenate([ mean_pa_0 , [gar_pa_0[1]] , [gar_pa_0[4]] , gar_pa_0[2:4]])
 #       start_params = np.array([ 0.201, 2.41, -0.157 , 0.00006 , 0.918 , 0.121, -0.043 ])                    
        return super(garch_m, self).fit(start_params=start_params, maxiter=maxiter, maxfun = maxfun, **kwds)

# %% table 5
data_5 = data[(data['year'] >= 1953) & (data['year'] <= 1984)]
model_garch_sp = garch_m(data_5['spread']*100)
results_g_sp = model_garch_sp.fit()
results_g_sp.summary()

# %%
class table_5(object):
    def __init__(self, data, year_comp):
        self.data = data
        self.year_comp = year_comp        
    def compt(self):
        model = arch_c(self.data['spread']*100)
        model_fit = model.fit()
        return model_fit   
    def table(self):
        t = self.compt()
        coef = np.array(t.params)
        H1 = np.array(["alpha", "theta", "a", "b", "c_1" , "c_2"])
        table = tabulate([coef], headers = H1, floatfmt=".4f") 
        return table    
    
    def table_comp_a(self):
        t = self.compt()
        year_comp = self.year_comp
        ind = 2
        if year_comp == 0:
            fre_p = []
            ind = 1
        elif year_comp == 1:
            fre_p = np.array([0.000324, -0.157, 0.00000062, 0.919, 0.121, -0.044])    
        elif year_comp == 2:
            fre_p = np.array([0.000496, -0.090, 0.00000149, 0.898, 0.106, -0.012])    
        else:
            fre_p = np.array([0.000257, -0.211, 0.00000052, 0.922, 0.130, -0.060])    
        H1 = np.array(["alpha", "theta", "a", "b", "c_1" , "c_2"])
        coef = np.array(t.params)        
        d_p = np.concatenate((fre_p,coef),axis=0).reshape(ind,-1)
        table = tabulate(d_p , headers = H1, floatfmt=".4f") 
        return table              
    
tab_5 = table_5(data, 3)
print(tab_5.table_comp_a())






# %% table 6 a 

model_garch_cr = garch_m(data_crsp[(data_crsp['year'] >= 1953)& (data_crsp['year'] <= 1984) ]['spread'])
results_g_cr = model_garch_cr.fit()
results_g_cr.summary()


# %%

from arch.univariate import ConstantMean, GARCH
gar_0 = ConstantMean(data['spread'])
gar_0.volatility = GARCH(p=2, q=1)
gar_0_r = gar_0.fit()
gar_pa_0 = np.array(gar_0_r.params)
# %%
sigma_2 = gar_0_r.conditional_volatility
X = sm.add_constant(sigma_2)         
#mean_0 = sm.tsa.ARMA(data['spread'], order=(0,1))
mean_0 = statsmodels.tsa.arima_model.ARMA(data['spread'],exog=sigma_2,order=(0,1))
mean_0_r = mean_0.fit()
mean_pa_0 = np.array(mean_0_r.params)
