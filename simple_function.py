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
# %%

from prettytable import PrettyTable
from prettytable import MSWORD_FRIENDLY


x = PrettyTable()
x.set_style(MSWORD_FRIENDLY)

x.field_names = ["theta_0", "theta_1", "theta_2", "theta_3"]
x.add_row(tab)
x.int_format = "%{0:3f}"

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


