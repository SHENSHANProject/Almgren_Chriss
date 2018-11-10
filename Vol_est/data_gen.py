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


#Generation of data using method in Almgren and Chriss with the concept of market noise
# dt = 1s
def Xt(x0,M=1,miu = 0.05,k = 5,alpha = 0.04,gamma = 0.5,rho =-0.5,T = 1.0/252,N = int(6.5*3600)+1):
    t = T/(N-1)
    x = np.zeros((M,N))
    sigma4 = np.zeros((M,))
    sigma2 = np.zeros((M,))
    
    for j in range(M):
        vi = 0.05
        x[j,0]=x0
        dsigma4 = 0# for calclating the integral of sigma 4
        dsigma2 = 0 #for calclating the integral of sigma 2
        for i in range(1,N):
            dB,dW = npr.multivariate_normal([0,0],[[t,rho*t],[rho*t,t]])
            x[j,i] = x[j,i-1]+(miu-vi/2)*t+np.sqrt(vi)*dB
            vi = vi+k*(alpha-vi)*t+gamma*np.sqrt(vi)*dW
            dsigma4 += vi**2*t
            dsigma2 += vi*t
        sigma4[j] = dsigma4
        sigma2[j] = dsigma2
    return x,sigma4,sigma2


#x = Xt(1,np.log(20))
#n = np.arange(int(6.5*3600)+1)
#s = np.exp(x)
#
#plt.plot(n,s.T)
# Real observation with noise
def Yt(M,xt,var_e = 0.0005**2):
    N = xt.shape[1]
    epsilon = np.random.randn(M,N)*np.sqrt(var_e)
    return xt+epsilon


# =============================================================================
# #Generate data with a deterministe volatility
# # Xt in incertitude region
# =============================================================================
def vol(T = 1.0/252,N = int(6.5*3600)+1):
    vmax = 0.015
    vmin = 0.005
    n_range = np.arange(N)
    vol = (vmax-vmin)/2*np.cos(n_range/(N-1)*2*np.pi-np.pi/8)+(vmax+vmin)/2
    
    return vol
    
# =============================================================================
# Visulisation of volatility in a day (6.5 hours trading time)
# vol = vol()
# plt.figure()
# plt.plot(vol)    
# =============================================================================

#output: array of dimension M*N M is the samples' number
def Xt_IR(M,x0,alpha,eta,T = 1.0/252,N = int(6.5*3600)+1,L=1):
    t = T/(N-1)
    x = np.zeros((M,N))
    sigma = vol()
    x[:,0]=int(x0/alpha)*alpha
    for i in range(M):
        for j in range(1,N):
            dw = npr.randn(1)*np.sqrt(t)
            x[i,j]=x[i,j-1]+sigma[j]*x[i,j-1]*dw
    for i in range(M):
        current_price = x0
        for j in range(1,N):
            if x[i,j] <= current_price-alpha*(L-1/2+eta):
                x[i,j] = current_price-alpha*L
            elif x[i,j]>= current_price+alpha*(L-1/2+eta):
                x[i,j] = current_price+alpha*L
            else:
                x[i,j] = current_price
            current_price = x[i,j]
    return x
    
# =============================================================================
#     
# xt = Xt_IR(1,100,alpha=0.05,eta=0.05)
# plt.figure()
# plt.plot(xt[0])
# plt.show()
#     
# =============================================================================
