

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import statsmodels.tsa.stattools as stats

from sklearn import ensemble
from sklearn import decomposition
from sklearn import feature_selection

def autocorrelation_numpy(time_series):

    #(define variable f here as your time series data)
    N = len(f)
    fvi = np.fft.fft(f, n=2*N)
    acf = np.real( np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
    acf = acf/N
    #To get the autocorrelation at lags 0 through M we then do:
    #autocorrelations at lags 0:M = acf[:M+1]
    #So element k in this vector is the autocorrelation at lag k (Python arrays start at index zero).
    return acf

def autocorrelation(time_series, lags, fft=True):
    
    acf = stats.acf(time_series, fft=fft)
    #acf = stats.acf(time_series, nlags=6000, qstat=True, fft=True)
    return acf

def crosscorrelation(time_series1, time_series2, unbiased):
    ccf = stats.ccf(time_series1, time_series2, unbiased=False)
    return ccf

def pca_analysis(trainX, components=2, verbose=False):
    """
    """
    pca = decomposition.PCA(n_components=components)
    X_r = pca.fit(trainX).transform(trainX)

    if verbose:
        print('explained variance ratio (first %s components): %s' % (str(components), str(pca.explained_variance_ratio_)))
        
    return X_r, pca.explained_variance_ratio_

def pca_plot(X_r, trainY, target_names, elev=-40, azim=-80):
    """
    """

    if len(target_names) != len(np.unique(trainY)):
        raise ValueErro("Target names and trainY categories must have the same lenght")

    fig = plt.figure()
    if(X_r.shape[1] == 1):
        plt.plot(X_r[trainY == i, 0], alpha=0.7, label=target_name)
    elif(X_r.shape[1] == 2):        
        for c, m, i, alpha, target_name in zip("grb", "DH+", [0, 1, 2], [0.8, 0.2, 0.5], target_names):            
            plt.scatter(X_r[trainY == i, 0], X_r[trainY  == i, 1], c=c, marker=m, alpha=alpha, label=target_name)
    elif(X_r.shape[1] == 3):
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elev, azim=azim)
        for c, m, i, alpha, target_name in zip("gr", "DH", [0, 1], [0.8, 0.2], target_names):            
            ax.scatter(X_r[trainY == i,0], X_r[trainY == i,1], X_r[trainY == i,2], c=c, marker=m, alpha=alpha, label=target_name)
    
    
    plt.legend()
    plt.title('PCA')
    plt.savefig('/Users/Pablo/Desktop/figure2.png')



def feature_importance(trainX, trainY, testX, testY, columns):
    """
        Calculates the feature importance on the training set for a given set of variables
        It prints this importance and plots it
        """
    
    ## Feature selection
    clf = ensemble.ExtraTreesClassifier(random_state=1729, n_estimators=250, n_jobs=-1)
    selector = clf.fit(trainX, trainY)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(trainX.shape[1]):
        print("%d. %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(trainX.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(trainX.shape[1]), indices)
    plt.xlim([-1, trainX.shape[1]])
    plt.show()


def feature_selection_trees(trainX, trainY, testX, testY):   
    """
    Calculate the feature importance and select the most importance features
    It return the filtered training and testing sets
    """
    ## Feature selection
    clf = ensemble.ExtraTreesClassifier(random_state=1729, n_estimators=250, n_jobs=-1)
    selector = clf.fit(trainX, trainY)

    fs = feature_selection.SelectFromModel(selector, prefit=True)
    trainX = fs.transform(trainX)
    testX = fs.transform(testX)

    return trainX, testX