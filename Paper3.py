#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 18:38:44 2018

@author: shenchali
"""
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

#simulation size and sampling precision
M = int(1e2)
N = int(1*8*3600*10)

#model paramaters
x0 = 100
delta_t = 1/N
alpha = 0.05
eta = 0.05
L = 1

def round_tick(x):
    return int(round(x/alpha))

def round_price(x):
    return int(round(x/alpha))*alpha

round_tick = np.vectorize(round_tick)
round_price = np.vectorize(round_price)

time = np.linspace(0,1,N+1)
efficient_price = np.zeros((M,N+1))
observed_price = np.zeros((M,N+1))
#we took the periodic function sin with period=1 as the volatility function
sigma_t = 0.005*np.sin(2*np.pi*(time+1/8))+0.01

efficient_price[:,0] = x0
W_t = np.sqrt(delta_t)*npr.normal(loc=0, scale=1,size=(M,N))

for i in range (1,N+1):
    efficient_price[:,i] = efficient_price[:,i-1] + sigma_t[i-1]*efficient_price[:,i-1]*W_t[:,i-1]

#simulate observed price using the model of uncertainty zone
observed_price[:,0] = round_tick(efficient_price[:,0])

for i in range(1,N+1):
    last_traded_price = alpha * observed_price[:,i-1]
    upper = last_traded_price + alpha*(L-1/2+eta)
    lower = last_traded_price - alpha*(L-1/2+eta)
    increment = (efficient_price[:,i]>=upper)
    decrement = (efficient_price[:,i]<=lower)
#    observed_price_mouvement[:,i-1] = increment*1 + decrement*(-1)
    observed_price[:,i] = (increment*L + observed_price[:,i-1])*increment + (decrement*(-L) + observed_price[:,i-1])*decrement + observed_price[:,i-1] * (1-increment-decrement)
#    observed_price[:,i] = observed_price[:,i-1] + observed_price_mouvement[:,i-1]*L
observed_price = observed_price*alpha

plt.plot(time,observed_price[67,:])
plt.plot(time,efficient_price[67,:])

def detect_jumping_point(X):
    jumping_points = []
    jumping_points_value = [X[0]]
    latest = X[0]
    for i in range(1,len(X)):
        if (X[i] > latest):
            jumping_points.append(1)
            jumping_points_value.append(X[i])
            latest = X[i]
        elif (X[i] < latest):
            jumping_points.append(-1)
            jumping_points_value.append(X[i])
            latest = X[i]
        else:
            continue
    return (jumping_points,jumping_points_value)
            
#the estimation of eta
alternation = np.zeros(M)
continuation = np.zeros(M)
for i in range(M):
#    jumping_point = observed_price_mouvement[i,:][observed_price_mouvement[i,:] != 0]
    jumping_point = detect_jumping_point(observed_price[i,:])[0]
    for j in range(1,len(jumping_point)):
        if (jumping_point[j]*jumping_point[j-1] == (-1)):
            alternation[i] += 1
        else:
            continuation[i] += 1
#    alternation += ((observed_price[:,i]-observed_price[:,i-1])*(observed_price[:,i-1]-observed_price[:,i-2]) < 0)
#    continuation += ((observed_price[:,i]-observed_price[:,i-1])*(observed_price[:,i-1]-observed_price[:,i-2]) > 0)

    

eta_estimated = continuation/(2*alternation)
print(np.mean(eta_estimated))
    
#efficient price retrieval and estimation of integrated volatility
X_retrieved = []
integrated_vol = []
for i in range(M):
    path_retrieved = [x0]
    jumping_price = detect_jumping_point(observed_price[i,:])[1]
    for j in range(1,len(jumping_price)):
        path_retrieved.append(jumping_price[j] - np.sign(jumping_price[j]-jumping_price[j-1])*(1/2-eta)*alpha)
    vol_estimator = 0
    for j in range(1,len(path_retrieved)):
        vol_estimator += np.square((path_retrieved[j]-path_retrieved[j-1])/path_retrieved[j-1])
    
    integrated_vol.append(vol_estimator)
    X_retrieved.append(path_retrieved)

plt.hist(integrated_vol)





