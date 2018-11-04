#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 21:36:37 2018

@author: shancheng
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy
from math import *

np.random.seed(10)
# M is the number of simulations we will perform
M= int(1e5)

N = 5 # Number of time periods
T = 5 # Time of liquidation
t = T/N # length of time intervel of two liquidations T/N

gamma = 2.5*1e-7
eta = 2.5*1e-6
epsilon = 0.0625

sigma_stock = 0.95
sigma_used = sigma_stock

alpha = 0.02

lamda = 1e-6

eta_til = eta*(1-gamma*t/2/eta)
k_til = np.sqrt(lamda*sigma_used*sigma_used/eta_til)
#solve k using solve function in sympy
k = sympy.symbols('k')
k = float(sympy.solve(sympy.cosh(k*t)-1-k_til*k_til*t*t/2,k)[1])


s0 = 50 # $ price of stock in the beginning
x0 = 1e6 # initial holdings


#Initialisation of our strategy
n = np.zeros(N+1)
X = np.zeros(N+1)
X[0] = x0
X = (np.sinh(k*(T-np.arange(N+1)))/np.sinh(k*T))*x0
n[1:]=X[0:-1]-X[1:]

plt.figure()
plt.scatter(np.arange(N+1),n)  


#permanent market impact function
def g(v):
    return gamma*v

#short-term market impact function
def h(t,x):
    return epsilon*np.sign(x)+ eta*x/t

#Initialisation of price of stock
S_Market_Price = np.zeros((M,N+1)) 
S_Market_Price[:,0] = s0

S_Execution_Price = np.zeros((M,N+1)) 
S_Execution_Price[:,0] = s0

for i in range(1,N+1):
    S_Market_Price[:,i] = S_Market_Price[:,i-1] + sigma_stock*np.sqrt(t)*np.random.normal(loc=0,scale=1,size=M) - t*g(n[i]/t)

for i in range(1,N+1):
    S_Execution_Price[:,i] = S_Market_Price[:,i-1] - h(t,n[i])

capture = S_Execution_Price.dot(n)
shortfall = x0*s0- capture

#plt.hist(shortfall,bins=3*int(M**(1/3)),normed=True)
#plt.hist(capture,bins=3*int(M**(1/3)),normed=True)

mean_shortfall = np.mean(shortfall)
var_shortfalll = np.var(shortfall)

utility = mean_shortfall+lamda*var_shortfalll

print("Sigma of stock is {}, sigma used is {}, our utility result is {}".format(sigma_stock,sigma_used,utility))



def final_utility(ratio):# this parameter equals to sigma_used/sigma_stock
    sigma_used = ratio * sigma_stock
    eta_til = eta*(1-gamma*t/2/eta)
    k_til = np.sqrt(lamda*sigma_used*sigma_used/eta_til)
    #solve k using solve function in sympy
    k = sympy.symbols('k')
    k = float(sympy.solve(sympy.cosh(k*t)-1-k_til*k_til*t*t/2,k)[1])

    #Initialisation of our strategy
    n = np.zeros(N+1)
    X = np.zeros(N+1)
    X[0] = x0
    X = (np.sinh(k*(T-np.arange(N+1)))/np.sinh(k*T))*x0
    n[1:]=X[0:-1]-X[1:]
    #Initialisation of price of stock
    S_Market_Price = np.zeros((M,N+1)) 
    S_Market_Price[:,0] = s0
    
    S_Execution_Price = np.zeros((M,N+1)) 
    S_Execution_Price[:,0] = s0
    
    for i in range(1,N+1):
        S_Market_Price[:,i] = S_Market_Price[:,i-1] + sigma_stock*np.sqrt(t)*np.random.normal(loc=0,scale=1,size=M) - t*g(n[i]/t)
    
    for i in range(1,N+1):
        S_Execution_Price[:,i] = S_Market_Price[:,i-1] - h(t,n[i])
    capture = S_Execution_Price.dot(n)
    shortfall = x0*s0- capture
    mean_shortfall = np.mean(shortfall)
    var_shortfalll = np.var(shortfall)

    utility = mean_shortfall+lamda*var_shortfalll
    
    return utility

ratio = np.arange(1,101)*0.1
utilities = np.zeros(len(ratio))
for i in range(len(ratio)):
    utilities[i] = final_utility(ratio[i])
plt.figure()
plt.plot(ratio,utilities)
plt.xlabel(r"$\frac{\sigma_{used}}{\sigma}$")
plt.ylabel("Final utility")
plt.show()





