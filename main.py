# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:37:47 2022

@author: Wu
"""
import pandas as pd
import numpy as np
import arch
import scipy.optimize as spo

df = pd.read_csv('data.csv', index_col=['timestampNano'], parse_dates=True)

weights = np.array([0.5]*4)

def sigma(t, params):
    ht = np.zeros_like(t)
    zt = np.zeros_like(t)
    
    ht[0] = 0.1
    zt[0] = (t[0] - params[0])/np.sqrt(ht[0])

    for i in range(1, len(t)):
        ht[i] = params[1] + (params[2] * np.square(t[i-1]-params[0])) + (params[3] * ht[i-1]) #+ (params[4] * xt[i-1])
        zt[i] = (t[i] - params[0])/np.sqrt(ht[i])
        
    return ht, zt

def objective(weights, returns):
    return -f(weights, returns)

def f(X, dataset):
    dataset['ht'], dataset['zt'] = sigma(dataset['pct'].values, X)
    dataset['L1'] = -0.5*(np.log(2*np.pi)+np.log(dataset['ht'])+np.square(dataset['zt']))
    return np.sum(dataset['L1'])

bnds = [(-1, 1)] * len(weights)    # https://stackoverflow.com/questions/29150064/python-scipy-indexerror-the-length-of-bounds-is-not-compatible-with-that-of-x0
best_mix = spo.minimize(objective, weights, args=(df), method='Nelder-Mead', bounds = bnds, options={'disp':False}) # , constraints = cons
print('mu, omega, alpha, beta:', best_mix.x,'\n') # SLSQP

am = arch.arch_model(df['pct'].values)
res = am.fit(update_freq=5)
print(res.params)