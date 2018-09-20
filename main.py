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
N = 10 # Nombre of time periods
T = 10 # Time of liquidation
t = 1 # length of time intervel of two liquidations T/N

gamma = 2.5*1e-7
eta = 2.5*1e-6
epsion = 0.0625

sigma = 0.95
alpha = 0.02

lamda = 1e-6

eta_til = eta*(1-gamma*t/2/eta)
k_til = sqrt(lamda*sigma*sigma/eta_til)
#solve k using solve function in sympy
k = sympy.symbols('k')
k = sympy.solve(sympy.cosh(k)-1-k_til*k_til*t*t/2,k)[1]


s0 = 50 # $ price of stock in the beginning
x0 = 1e6 # initial holdings


#Initialisation of our strategy
n = np.zeros(T+1)
X = np.zeros(T+1)
X[0] = x0
X = (np.sinh((k*(T-np.arange(T+1))).astype(float))/(sympy.sinh((k*T)))).astype(float)*x0
n[1:]=X[0:-1]-X[1:]

#plt.scatter(np.arange(T+1),n)
#plt.close()

def g(v):
    return gamma*v

def h(t,nk):
    return epsilon*np.sign(nk)+eta/t*nk

np.random.seed(8)

#Initialisation of price of stock
S = np.zeros(T+1) 
S[0] = s0










