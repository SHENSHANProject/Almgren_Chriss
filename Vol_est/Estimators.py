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

def Est3(xt,n_sparse):
    M,N = xt.shape
    res = np.zeros((M,))
    for i in range(M):
        sample = xt[i,::n_sparse[i]]
        ds =sample[1:]-sample[:-1]
        res[i] = np.sum(ds**2)
        
    return res

def Est2(xt,n_avg_opt):
    M,N = xt.shape
    res = np.zeros((M,))
    for j in range(M):
        K = int(N/n_avg_opt[j])
        nest = np.zeros(K)
       
        for i in range(K):
            sample = xt[j,i::K]
            ds =sample[1:]-sample[:-1]
            
            nest[i] = np.sum(ds**2)
        res[j] = np.sum(nest)/K
    
    return res

def Est1(xt,n_avg_opt):
    E5 = Est5(xt)
    E2 = Est2(xt,n_avg_opt)
    M,N = xt.shape
    
    res = np.zeros((M,))
    for i in range(M):
        res[i] = E2[i] - n_avg_opt[i]/N * E5[i]
    return res
def n_opt_sparse(T,N,est5,sigma4):
    n = (T*sigma4/(est5/(2*N))**2/6)**(1/3)
    
    return n.astype(int)

def n_opt_avg(T,N,est5,sigma4):
    n = (T*sigma4/(est5/(2*N))**2/4)**(1/3)
    
    return n.astype(int)
def QV(xt):
    M,N = xt.shape
    dx = xt[:,1:]-xt[:,:-1]
    res = np.sum(dx**2,axis =1)
    
    return res

def MSE(x,y):
    return np.sum((x-y)**2)/len(x)

#if __name__=="__main__":
T = 1.0/252
N = int(6.5*3600)+1
M = 10000
x0 = np.log(25)
xt,sigma4,sigma2= Xt(x0)
yt = Yt(M,xt)

Qv_xt = QV(xt) # Quadratic variation of Xt (latent price)

est5 = Est5(yt)
#est4 = Est4(yt)

n1 = n_opt_sparse(T,N,est5,sigma4)
n2 = n_opt_avg(T,N,est5,sigma4)

est1 = Est1(yt, n2)

#print("mean of n {}".format(np.mean(n)))
#print("RMSE of Est5{:.4E}".format(np.sqrt(MSE(Qv_xt,est5))))
#print("RMSE of Est4{:.4E}".format(np.sqrt(MSE(Qv_xt,est4))))
#

plt.figure()
plt.hist(est1-sigma2)

plt.show()



