# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:56:56 2018

@author: s3179575
"""
import numpy as np
import math

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