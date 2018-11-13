# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:32:29 2018

@author: s3179575
"""

	
# example of ARCH model
from random import gauss
from random import seed
from matplotlib import pyplot
from arch import arch_model
# seed pseudorandom number generator
seed(1)
# create dataset
data = [gauss(0, i*0.01) for i in range(0,100)]
# split into train/test
n_test = 10
train, test = data[:-n_test], data[-n_test:]
# define model
model = arch_model(train, mean='Zero', vol='ARCH', p=1)
# fit model
model_fit = model.fit()
model_fit.summary()
# forecast the test set
#yhat = model_fit.forecast(horizon=n_test)
# plot the actual variance
#var = [i*0.01 for i in range(0,100)]
#pyplot.plot(var[-n_test:])
# plot forecast variance
#pyplot.plot(yhat.variance.values[-1, :])
#pyplot.show()



# %%

import datetime as dt

import pandas_datareader.data as web

from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal
#from arch.univariate import ZeroMean, GARCH, Normal

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2014, 1, 1)
sp500 = web.DataReader('^GSPC', 'yahoo', start=start, end=end)
returns = 100 * sp500['Adj Close'].pct_change().dropna()

am = ConstantMean(returns)
am.volatility = GARCH(1, 0, 1)
am.distribution = Normal()

res = am.fit()

res.summary()

# %%

# import the packages
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
import time

# Set up your x values
x = np.linspace(0, 100, num=100)

# Set up your observed y values with a known slope (2.4), intercept (5), and sd (4)
yObs = 5 + 2.4*x + np.random.normal(0, 4, 100)

# Define the likelihood function where params is a list of initial parameter estimates
def regressLL(params):
    # Resave the initial parameter guesses
    b0 = params[0]
    b1 = params[1]
    sd = params[2]

    # Calculate the predicted values from the initial parameter guesses
    yPred = b0 + b1*x

    # Calculate the negative log-likelihood as the negative sum of the log of a normal
    # PDF where the observed values are normally distributed around the mean (yPred)
    # with a standard deviation of sd
    logLik = -np.sum( stats.norm.logpdf(yObs, loc=yPred, scale=sd) )

    # Tell the function to return the NLL (this is what will be minimized)
    return(logLik)

# Make a list of initial parameter guesses (b0, b1, sd)    
initParams = [1, 1, 1]

# Run the minimizer
results = minimize(regressLL, initParams, method='nelder-mead')

# Print the results. They should be really close to your actual values
print(results.x)

# %%
from __future__ import division

from matplotlib import  pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.base.model import GenericLikelihoodModel

np.random.seed(123456789)
pi = 0.3
lambda_ = 2.

def zip_pmf(x, pi=pi, lambda_=lambda_):
    if pi < 0 or pi > 1 or lambda_ <= 0:
        return np.zeros_like(x)
    else:
        return (x == 0) * pi + (1 - pi) * stats.poisson.pmf(x, lambda_)
    
# we generate 1,000 observations from the zero-inflated model.   
N = 1000
inflated_zero = stats.bernoulli.rvs(pi, size=N)
x = (1 - inflated_zero) * stats.poisson.rvs(lambda_, size=N)

# We are now ready to estimate π and λ by maximum likelihood. To do so, we define a class that inherits from statsmodels’ GenericLikelihoodModel as follows.

class ZeroInflatedPoisson(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(ZeroInflatedPoisson, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
        pi = params[0]
        lambda_ = params[1]

        return -np.log(zip_pmf(self.endog, pi=pi, lambda_=lambda_))
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            lambda_start = self.endog.mean()
            excess_zeros = (self.endog == 0).mean() - stats.poisson.pmf(0, lambda_start)
            
            start_params = np.array([excess_zeros, lambda_start])
            
        return super(ZeroInflatedPoisson, self).fit(start_params=start_params,
                                                    maxiter=maxiter, maxfun=maxfun, **kwds)

model = ZeroInflatedPoisson(x)
results = model.fit()

    

# %% Probit model

from __future__ import print_function
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

data = sm.datasets.spector.load_pandas()
exog = data.exog
endog = data.endog
print(sm.datasets.spector.NOTE)
print(data.exog.head())

exog = sm.add_constant(exog, prepend=True)

class MyProbit(GenericLikelihoodModel):
    def loglike(self, params):
        exog = self.exog
        endog = self.endog
        q = 2 * endog - 1
        return stats.norm.logcdf(q*np.dot(exog, params)).sum()
    
sm_probit_manual = MyProbit(endog, exog).fit()
print(sm_probit_manual.summary())

 # Compare your Probit implementation to statsmodels' "canned" implementation:

sm_probit_canned = sm.Probit(endog, exog).fit()
    


# %%

import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
import math 

# (1/np.sqrt(scale*2*math.pi))*np.exp(-0.5*np.power((1-loc)/scale,2))
    
# Estimation of a GARCH model by ML.

def cun_pmf(x, rho_0, rho_1, rho_2 , mu):
    tst = (1 - rho_1 - rho_2)
    if tst != 0:
        ivar = rho_0/(1 - rho_1 - rho_2)
    else:
        ivar = 0.01
    # compute residuals
    #e = x - mu

    n = len(x)
    e = np.zeros(n)    
    ss = np.sum(e*e)/(n-4)
    
    # generate sigt and log likelihood
    sigt = np.zeros(n);
    loglik = np.zeros(n);

    for i in range(n):
        if i == 0:
            sigt[0] = rho_0 + rho_1*ss + rho_2*ss;
            e[0] = x[0] - mu*np.sqrt(sigt[0])
        else:
            
            sigt[i] = rho_0 + rho_1*sigt[i-1] + rho_2*e[i-1]*e[i-1];
            e[i] = x[i] - mu*np.sqrt(sigt[i])
        
        loglik[i] = -0.5*(np.log(2*math.pi) + np.log(sigt[i]) + (e[i]*e[i])/sigt[i]);
    
    #if rho_0 < 0 or rho_1 < 0 :
    #    return np.zeros_like(x)
    #else:
    #    LogL = 0
    #    for i in range(1,len(x)):
    #        std =  rho_0 + rho_1*np.power(x[i-1],2)
    #        LL = np.power(x[i],2)/std + np.log(std)
    #        LogL = LogL + LL
            
    return -np.sum(loglik)

# %%
class arch_c(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(arch_c, self).__init__(endog, exog, **kwds)
    
    def nloglikeobs(self, params):
       # mean = 0  params[0] +
        rho_0 = params[0]
        rho_1 = params[1]
        rho_2 = params[2]
        mu = params[3]
        #std = params[0] + params[1]*self.endog

      #  return -np.log( stats.norm.pdf(self.endog, mean, std))
      #  return np.power(self.endog,2)/np.power(std,2) + np.log(np.power(std,2))
        return cun_pmf(self.endog,rho_0,rho_1,rho_2,mu)
      #  return LogL
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        #if start_params is None:
        #    lambda_start = self.endog.mean()
        #    excess_zeros = (self.endog == 0).mean() - stats.poisson.pmf(0, lambda_start)
            
        start_params = np.array([0.1,0.1, 0.5,1])
            
        return super(arch_c, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)

model = arch_c(returns)
results = model.fit()
results.summary()

# %%

for i in range(len(data)):
     if i == 0:
       print(data[i])
       print('value',i)
     else:
       print(data[i])         
       print(i)
    
    