#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:39:36 2018

@author: shancheng
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
#miu = 0.05
#k = 5
#alpha = 0.04
#gamma = 0.5
#rho =-0.5
#
#T = 1.0/252 # T= 1 day
## each day 6.5h open trading
#N = int(6.5*3600)+1 # number of each simulation 
#t = T/N # time interval to collecting data equals 1 second

# Parameter for noise
#var_e = 0.0005**2 # variance of noise

def Xt(M,x0,miu = 0.05,k = 5,alpha = 0.04,gamma = 0.5,rho =-0.5,T = 1.0/252,N = int(6.5*3600)+1):
    t = T/N
    x = np.zeros((M,N))
    
    
    for j in range(M):
        vi = 0.05
        x[j,0]=x0
        for i in range(1,N):
            dB,dW = npr.multivariate_normal([0,0],[[t,rho*t],[rho*t,t]])
            x[j,i] = x[j,i-1]+(miu-vi/2)*t+vi**2*dB
            vi = vi+k*(alpha-vi)*t+gamma*np.sqrt(vi)*dW
    
    return x


#x = Xt(1,np.log(20))
#n = np.arange(int(6.5*3600)+1)
#s = np.exp(x)
#
#plt.plot(n,s.T)

def Yt(xt,var_e = 0.0005**2):
    M = xt.shape[0]
    N = xt.shape[1]
    epsilon = np.random.randn(M,N)*np.sqrt(var_e)
    return xt+epsilon