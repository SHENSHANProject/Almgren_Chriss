#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 21:36:37 2018

@author: shancheng
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import sympy
from math import *
<<<<<<< HEAD
N = 10 # Nombre of time periods
T = 10 # Time of liquidation
=======
# M is the number of simulations we will perform
M= int(1e5)

N = 5 # Number of time periods
T = 5 # Time of liquidation
>>>>>>> 3f1b5d1546cc5710156641d67c9f22638e1f44e8
t = T/N # length of time intervel of two liquidations T/N

gamma = 2.5*1e-7
eta = 2.5*1e-6
epsilon = 0.0625

sigma_stock = 0.95
sigma_used = sigma_stock/10

<<<<<<< HEAD
sigma = 9.5
=======
>>>>>>> 3f1b5d1546cc5710156641d67c9f22638e1f44e8
alpha = 0.02

sigma_real = 0.95

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
<<<<<<< HEAD
X = (np.sinh((k*(T-np.arange(N+1)*t)).astype(float))/(sympy.sinh((k*T)))).astype(float)*x0
n[1:]=X[0:-1]-X[1:]

plt.scatter(np.arange(N+1),n)
plt.show()
=======
X = (np.sinh(k*(T-np.arange(N+1)))/np.sinh(k*T))*x0
n[1:]=X[0:-1]-X[1:]

plt.scatter(np.arange(T+1),n)
plt.figure()
>>>>>>> 3f1b5d1546cc5710156641d67c9f22638e1f44e8

#permanent market impact function
def g(v):
    return gamma*v

<<<<<<< HEAD
def h(t,nk):
    return epsion*np.sign(nk)+eta/t*nk

#Initialisation of price of stock
S = np.zeros(N+1) 
S[0] = s0

V = np.zeros(1000)
for i in range(1000):
    xi_normal = npr.normal(0,1,N)
    S[1:]=s0+sigma_real*np.sqrt(t)*np.cumsum(xi_normal)-gamma*(x0-X[1:])
    
    S_tilde = S.copy()
    S_tilde[1:] = S[0:-1]-np.vectorize(h)(t,n[1:])
    
    #Total liquidation
    V[i] = np.dot(n,S_tilde)

V_mean = np.sum(V)/len(V)
print("Mean of liquidation value {}".format(V_mean))

=======
#short-term market impact function
def h(t,x):
    return epsilon*np.sign(x)+ eta*x/t

#Initialisation of price of stock
S_Market_Price = np.zeros((M,T+1)) 
S_Market_Price[:,0] = s0

S_Execution_Price = np.zeros((M,T+1)) 
S_Execution_Price[:,0] = s0
>>>>>>> 3f1b5d1546cc5710156641d67c9f22638e1f44e8

for i in range(1,T+1):
    S_Market_Price[:,i] = S_Market_Price[:,i-1] + sigma_stock*np.sqrt(t)*np.random.normal(loc=0,scale=1,size=M) - t*g(n[i]/t)

for i in range(1,T+1):
    S_Execution_Price[:,i] = S_Market_Price[:,i-1] - h(t,n[i])

capture = S_Execution_Price.dot(n)
shortfall = x0*s0- capture

plt.hist(shortfall,bins=3*int(M**(1/3)),normed=True)
plt.hist(capture,bins=3*int(M**(1/3)),normed=True)





