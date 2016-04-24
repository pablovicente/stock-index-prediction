#!/usr/bin/python

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as stats

def autocorrelation_numpy(time_series):

    #(define variable f here as your time series data)
    N = len(f)
    fvi = np.fft.fft(f, n=2*N)
    acf = np.real( np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
    acf = acf/N
    #To get the autocorrelation at lags 0 through M we then do:
    #autocorrelations at lags 0:M = acf[:M+1]
    #So element k in this vector is the autocorrelation at lag k (Python’s arrays start at index zero).
    return acf

def autocorrelationf(time_series, lags, fft):
    acf = stats.acf(time_series, nlags=lags, fft=fft)
    return acf

def crosscorrelation(time_series1, time_series2, unbiased):
    ccf = stats.ccf(time_series1, time_series2, unbiased=False)
    return ccf