# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:56:56 2018

@author: s3179575
"""
import numpy as np
import math
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format

arg1 = np.array([-0.159 , 0.073, -0.157, 0.063, 0.918, 0.121, -0.043])

x = np.array(data['spread']*100)
coef = arg1

mu = coef[0]
beta = coef[1]
theta = coef[2]
a = coef[3]
b = coef[4]
c_1 = coef[5]
c_2 = coef[6]
#    [-0.001, 0.073, -0.157, 6e-05, 0.918, 0.121, -0.157]
    
    # compute residuals
n = len(x)
ss = np.std(x)
e = np.zeros(n)    
    # generate sigt and log likelihood
sigt = np.zeros(n)  ;
loglik = np.zeros(n);

for i in range(n):
        if i <= 1:
            sigt[i] = a + b*ss;
            e[i] = x[i] - mu - theta*ss 
            loglik[i] = -0.5*( np.log(2*math.pi) + np.log(sigt[i]) + (e[i]*e[i])/sigt[i] )            
        else:
            sigt[i] = a + b*sigt[i-1] + c_1*e[i-1]*e[i-1] + c_2*e[i-2]*e[i-2] 
            e[i] = x[i] - mu - beta*np.sqrt(sigt[i]) + theta*e[i-1]    
            loglik[i] = -0.5*( np.log(2*math.pi) + np.log(sigt[i]) + (e[i]*e[i])/sigt[i] )
                
print(np.sum(loglik))
# %%

from prettytable import PrettyTable
from prettytable import MSWORD_FRIENDLY
from tabulate import tabulate

x = PrettyTable()
x.set_style(MSWORD_FRIENDLY)

x.field_names = ["theta_0", "theta_1", "theta_2", "theta_3"]
x.add_row(tab)
#x.int_format = ".3f"

#print(x)    


H1 = np.array(["theta_0", "theta_1", "theta_2", "theta_3"])
fre_p = np.array([0.0, 0.524, 0.158, 0.09])
data = np.concatenate((tab,fre_p),axis=0).reshape(2,-1)
table = tabulate(data , headers = H1, floatfmt=".4f") 
print(table)
 

# https://github.com/vishvananda/prettytable
# "%<{0:3f}>d

# %%
x.field_names = ["City name", "Area", "Population", "Annual Rainfall"]
x.add_row(["Adelaide", 1295, 1158259, 600.5])
x.add_row(["Brisbane", 5905, 1857594, 1146.4])
x.add_row(["Darwin", 112, 120900, 1714.7])
x.add_row(["Hobart", 1357, 205556, 619.5])
x.add_row(["Sydney", 2058, 4336374, 1214.8])
x.add_row(["Melbourne", 1566, 3806092, 646.9])
x.add_row(["Perth", 5386, 1554769, 869.4])

print(x)    

# %%

# define the function blocks
def zero():
    print( "You typed zero.\n")

def sqr():
    print ("n is a perfect square\n")

def even():
    print("n is an even number\n")

def prime():
    print( "n is a prime number\n")

# map the inputs to the function blocks
options = {0 : zero,
           1 : sqr,
           4 : sqr,
           9 : sqr,
           2 : even,
           3 : prime,
           5 : prime,
           7 : prime,
}



class Switcher(object):
    def numbers_to_months(self, argument):
        """Dispatch method"""
        method_name = 'month_' + str(argument)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid month")
        # Call the method as we return it
        return method()
 
    def month_1(self):
        return "January"
 
    def month_2(self):
        return "February"
 
    def month_3(self):
        return "March"

# %%
# define the Vehicle class
class Vehicle:
    name = ""
    kind = "car"
    color = ""
    value = 100.00
    
    def description(self):
        desc_str = "%s is a %s %s worth $%.2f." % (self.name, self.color, self.kind, self.value)
        return desc_str

# your code goes here
car1 = Vehicle()
car1.name = "Fer"
car1.color = "red"
car1.kind = "convertible"
car1.value = 60000.00

# test code
print(car1.description())


# %%

class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age
    
  def myfunc(self):
    print("Hello my name is " + self.name)
    
p1 = Person("John", 36)

print(p1.name)
print(p1.age)
 
# %%
# http://hplgit.github.io/primer.html/doc/pub/class/._class-solarized003.html

class Derivative(object):
    def __init__(self, f, h=1E-5):
        self.f = f
        self.h = float(h)

    def __call__(self, x):
        f, h = self.f, self.h      # make short forms
        return (f(x+h) - f(x))/h   



# %%
import scipy.stats as sc

std = 4   
mean = 1.5

x_1 = 1
print(sc.norm(mean, std).pdf(x_1))    

pro = (1/(np.sqrt(2*math.pi*np.power(std,2))))*np.exp(-0.5*np.power((x_1-mean)/std,2))
print(pro)


# %%    
import math
from statsmodels.base.model import GenericLikelihoodModel
import time
import numpy as np
N = 500
x = np.ones(N)
std = np.exp( 1.8*(sc.norm(1,0.1).rvs(size=N)) )

y = x*2 + 3.5*np.power(std,2) + sc.norm(0,std).rvs(size=N)
#y = x*3.2 + 2*std + sc.norm(0,std).rvs(size=N)

def cun_pmf(x, mu, sig ,beta):
    # compute residuals
    n = len(x)
    e = np.zeros(n)    
    ss = np.sum(e*e)/(n-2)
    # generate sigt and log likelihood
    sigt = np.zeros(n);
    loglik = np.zeros(n);

    for i in range(n):
#        if i <= 1:
            sigt[i] = np.exp(2*sig);
            e[i] = x[i] - mu - beta*np.sqrt(sigt[i]) 
            
#            e[i] = x[i] - mu - beta*np.power(sigt[i],2)             
#        else:
            #sigt[i] = a + b*sigt[i-1] + c_1*e[i-1]*e[i-1] + c_2*e[i-2]*e[i-2];
#            sigt[i] = 
#            e[i] = x[i] - mu
            
 #           loglik[i]  = np.log( sc.norm(0, np.sqrt(sigt[i])).pdf(e[i]))
#            loglik[i] = -0.5*(np.log(2*math.pi) + np.log(np.power( sigt[i],2) ) + (e[i]*e[i])/(np.power( sigt[i],2) ));
 
#            loglik[i] = -0.5*(np.log(2*math.pi) + np.log(sigt[i]) + np.power(x[i]-mu-1,2) -(2*(x[i]-mu)+np.power(x[i]-mu,2))*(sigt[i]-1) -(2*(x[i]-mu))*(beta));                 
            loglik[i] = -0.5*(np.log(2*math.pi) + np.log(sigt[i]) + (e[i]*e[i])/( sigt[i] ));                
            
    return -np.sum(loglik)

class arch_c(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(arch_c, self).__init__(endog, exog, **kwds)
        
    def nloglikeobs(self, params):
        mu = params[0]
        sig = params[1]       
        beta = params[2]        
        
        return cun_pmf(self.endog,mu,sig,beta)
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        start_params = np.array([2.8, 1.05, 1.8])
            
        return super(arch_c, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)


model = arch_c(y)
model_fit = model.fit()
print(model_fit.summary())

# %%

def cun_1(x, mu, sig ,beta):
    # compute residuals
    n = len(x)
    e = np.zeros(n)    
    ss = np.sum(e*e)/(n-2)
    # generate sigt and log likelihood
    sigt = np.zeros(n);
    loglik = np.zeros(n);

    for i in range(n):
#        if i <= 1:
            sigt[i] = sig;
            e[i] = x[i] - mu - beta*np.sqrt(sigt[i]) 
            
#            e[i] = x[i] - mu - beta*np.power(sigt[i],2)             
#        else:
            #sigt[i] = a + b*sigt[i-1] + c_1*e[i-1]*e[i-1] + c_2*e[i-2]*e[i-2];
#            sigt[i] = 
#            e[i] = x[i] - mu
            
 #           loglik[i]  = np.log( sc.norm(0, np.sqrt(sigt[i])).pdf(e[i]))
#            loglik[i] = -0.5*(np.log(2*math.pi) + np.log(np.power( sigt[i],2) ) + (e[i]*e[i])/(np.power( sigt[i],2) ));
 
#            loglik[i] = -0.5*(np.log(2*math.pi) + np.log(sigt[i]) + np.power(x[i]-mu-1,2) -(2*(x[i]-mu)+np.power(x[i]-mu,2))*(sigt[i]-1) -(2*(x[i]-mu))*(beta));                 
            loglik[i] = -0.5*(np.log(2*math.pi) + np.log(sigt[i]) + (e[i]*e[i])/( sigt[i] ));                
            
    return -np.sum(loglik)

class arch_c1(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(arch_c1, self).__init__(endog, exog, **kwds)
        
    def nloglikeobs(self, params):
        mu = params[0]
        sig = params[1]       
        beta = params[2]        
        
        return cun_1(self.endog,mu,sig,beta)
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        start_params = np.array([2.8, 1.05, 1.8])
            
        return super(arch_c1, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)


t = time.time()

y_s = y/np.exp(2)

model = arch_c1(y_s)
model_fit = model.fit()
print(model_fit.summary())

elapsed = time.time() - t
# 0.24