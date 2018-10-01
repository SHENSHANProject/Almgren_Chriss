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
N = 10 # Nombre of time periods
T = 10 # Time of liquidation
t = T/N # length of time intervel of two liquidations T/N

gamma = 2.5*1e-7
eta = 2.5*1e-6
epsion = 0.0625

sigma = 9.5
alpha = 0.02

sigma_real = 0.95

lamda = 1e-6

eta_til = eta*(1-gamma*t/2/eta)
k_til = sqrt(lamda*sigma*sigma/eta_til)
#solve k using solve function in sympy
k = sympy.symbols('k')
k = sympy.solve(sympy.cosh(k)-1-k_til*k_til*t*t/2,k)[1]


s0 = 50 # $ price of stock in the beginning
x0 = 1e6 # initial holdings


#Initialisation of our strategy
n = np.zeros(N+1)
X = np.zeros(N+1)
X[0] = x0
X = (np.sinh((k*(T-np.arange(N+1)*t)).astype(float))/(sympy.sinh((k*T)))).astype(float)*x0
n[1:]=X[0:-1]-X[1:]

plt.scatter(np.arange(N+1),n)
plt.show()

def g(v):
    return gamma*v

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










