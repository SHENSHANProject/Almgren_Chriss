#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:56:08 2018

@author: shancheng
"""

from data_gen import Xt,Yt
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


np.random.seed(10)

def Est5(xt):
    M,N = xt.shape
    dx = xt[:,1:]-xt[:,:-1]
    res = np.sum(dx**2,axis =1)
    
    return res

def Est4(xt,K=5*60): # K is sampling frequency, here we take every 5 minutes so corresponds to every 5*60 data we take it as a smaple
    M,N = xt.shape
    sample = xt[:,::K]
    ds =sample[:,1:]-sample[:,:-1]
    
    res = np.sum(ds**2,axis=1)
    
    return res

def Est3(xt,K):
    return Est4(xt,K=K)

def Est2(xt,K):
    nest = np.zeros(K)
    M,N = xt.shape
    for i in range(K):
        sample = xt[:,i::K]
        ds =sample[:,1:]-sample[:,:-1]
        
        nest[i] = np.sum(ds**2,axis=1)
    res = np.sum(nest)/K
    
    return res

def Est1(xt,K):
    E5 = Est5(xt)
    E2 = Est2(xt,K)
    M,N = xt.shape
    n = N//K
    res = E2 - n/N * E5
    
    return res

def QV(xt):
    M,N = xt.shape
    dx = xt[:,1:]-xt[:,:-1]
    res = np.sum(dx**2,axis =1)
    
    return res

def MSE(x,y):
    return np.sum((x-y)**2)/len(x)

if __name__=="__main__":
    M = 10
    x0 = np.log(25)
    xt = Xt(M,x0)
    yt = Yt(xt)
    
    Qv_xt = QV(xt) # Quadratic variation of Xt (latent price)
    
    est5 = Est5(yt)
    est4 = Est4(yt)
    print("RMSE of Est5{:.4E}".format(np.sqrt(MSE(Qv_xt,est5))))
    print("RMSE of Est4{:.4E}".format(np.sqrt(MSE(Qv_xt,est4))))
    
    



