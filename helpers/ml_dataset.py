#!/usr/bin/python

import calendar
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.tsa.stattools as stats

from sklearn import feature_selection
from sklearn import ensemble

def generate_df_dataset(names_list, df_list, dfs_cols_dict):
    """

    Parameters
    --------------------
    - names_list: a list contaning the name of each dataframe as string
    - df_list: a list with the dataframes in the same order than the previous lits
    - dfs_cols_dict: a dictionary where the key is a df name and the values are the colums to process
    """
    
    rows, cols =  df_list[0].shape
    dates = df_list[0]['Date'].tolist()
    
    dict_values = {'Date':dates}
    dict_dfs_names = {}
    col_list = []
    
    for index in range(len(names_list)):
        name = names_list[index]
        df = df_list[index]
        
        cols = dfs_cols_dict[name]               
        new_cols = [name+"_"+x for x in cols]    
        col_list.extend(new_cols)
        
        for c, nc in zip(cols, new_cols):
            dict_values[nc] = df[c]        
    
    temp_names_list = list(col_list)
    temp_names_list.insert(0, 'Date')
    df_dataset = pd.DataFrame(dict_values)
    
    #Sort rows 
    df_dataset = df_dataset[temp_names_list]

    return df_dataset


def remove_columns_from_dataset(dataset, predicting='close', shifted = False):
    """
    """

    ###########################
    # predicting close price: #
    ###########################        
    colsToRemove = []
    colsToShift = []
    
    if predicting == 'close':
        if not shifted:        
            colsToRemove.extend([col for col in dataset.columns if 'Date' in col])
            colsToRemove.extend([col for col in dataset.columns if 'GOLD' in col and not '_AM' in col])
            colsToRemove.extend([col for col in dataset.columns if 'SILVER' in col and not '_USD' in col])
            colsToRemove.extend([col for col in dataset.columns if 'OIL_BRENT' in col and not '_USD' in col])
            colsToRemove.extend([col for col in dataset.columns if 'PLAT' in col and not '_AM' in col])
            colsToRemove.extend([col for col in dataset.columns if 'DJIA' in col and not '_Open' in col])
            colsToRemove.extend([col for col in dataset.columns if 'HSI' in col and '_Date' in col])
            colsToRemove.extend([col for col in dataset.columns if 'IBEX' in col and not '_Open' in col])
            colsToRemove.extend([col for col in dataset.columns if 'N225' in col and'_Date' in col])
            colsToRemove.extend([col for col in dataset.columns if 'SP500' in col and not '_Open' in col])
            colsToRemove.remove('IBEX_RD_B1_Close')            
        else:
            colsToRemove.extend([col for col in dataset.columns if 'Date' in col])            
            colsToShift.extend([col for col in dataset.columns if 'GOLD' in col and not '_AM' in col])
            colsToShift.extend([col for col in dataset.columns if 'SILVER' in col and not '_USD' in col])
            colsToShift.extend([col for col in dataset.columns if 'OIL_BRENT' in col and not '_USD' in col])
            colsToShift.extend([col for col in dataset.columns if 'PLAT' in col and not '_AM' in col])
            colsToShift.extend([col for col in dataset.columns if 'DJIA' in col and not '_Open' in col])
            colsToShift.extend([col for col in dataset.columns if 'HSI' in col and '_Date' in col])
            colsToShift.extend([col for col in dataset.columns if 'IBEX' in col and not '_Open' in col])
            colsToShift.extend([col for col in dataset.columns if 'N225' in col and'_Date' in col])
            colsToShift.extend([col for col in dataset.columns if 'SP500' in col and not '_Open' in col])
            colsToShift.remove('IBEX_RD_B1_Close')  

    ###########################
    # predicting open price: #
    ###########################

    if predicting == 'open' and not shifted:            
            colsToRemove.extend([col for col in dataset.columns if 'Date' in col])
            colsToRemove.extend([col for col in dataset.columns if 'GOLD' in col])
            colsToRemove.extend([col for col in dataset.columns if 'SILVER' in col])
            colsToRemove.extend([col for col in dataset.columns if 'PLAT' in col])
            colsToRemove.extend([col for col in dataset.columns if 'OIL_BRENT' in col])    
            colsToRemove.extend([col for col in dataset.columns if 'DJIA' in col])
            colsToRemove.extend([col for col in dataset.columns if 'HSI' in col and '_Open' in col])
            colsToRemove.extend([col for col in dataset.columns if 'IBEX' in col])
            colsToRemove.extend([col for col in dataset.columns if 'N225' in col and'_Date' in col])
            colsToRemove.extend([col for col in dataset.columns if 'SP500' in col])
            colsToRemove.remove('IBEX_RD_B1_Open')

    colsToShift = list(set(colsToShift) - set(colsToRemove)) 
    df = dataset.drop(colsToRemove, axis = 1)
    if shifted: 
        df[colsToShift] = df[colsToShift].shift(1)
        df = df[1:]
        df = df.reset_index(drop=True)
    
    return df


def dataset_to_train(train_df, test_df, predicting='close', binary = False, shifted = False):
    """
    
    Parameter
    ---------------------
    - train_df: dataframe containing the training rows and all features
    - test_df: dataframe containing the testing rows and all features
    - namesToRemove: partial names to search to remove those columns
    - colY: name of target
    - binary: boolean detemines whether features will be binary only 
    - shifted: boolean detemines whether rows will be shifted by one

    """
    colY = ''
    colsToRemove = []

    if predicting == 'close': 
        colY = 'IBEX_RD_B1_Close'
        colsToRemove = ['IBEX_RD_B1_Close']
    if predicting == 'open': 
        colY = 'IBEX_RD_B1_Open'
        colsToRemove = ['IBEX_RD_B1_Open']

    if(binary):
        colsToRemove.extend([col for col in train_df.columns if '_B' not in col])
        trainX = np.nan_to_num(np.asarray(train_df.drop(colsToRemove, axis = 1)))
        testX = np.nan_to_num(np.asarray(test_df.drop(colsToRemove, axis = 1)))
    else:
        trainX = np.nan_to_num(np.asarray(train_df.drop(colsToRemove, axis = 1)))
        testX = np.nan_to_num(np.asarray(test_df.drop(colsToRemove, axis = 1)))

    if shifted:
        trainY = np.nan_to_num(np.asarray(train_df[colY].shift(1)))
        testY = np.nan_to_num(np.asarray(test_df[colY].shift(1)))
    else:
        trainY = np.nan_to_num(np.asarray(train_df[colY]))
        testY = np.nan_to_num(np.asarray(test_df[colY]))

    return trainX, trainY, testX, testY

def dataset_to_train_using_dates(dataset, trainDates, testDates, predicting = 'close', binary = False, shiftFeatures = False, shiftTarget = False):
    """
    
    Parameter
    ---------------------
    - dataset: dataframe containing all available columns for a set of dates
    - trainDates: list containing the start training day and end training day
    - testDates: list containing the start training day and end testing day
    - namesToRemove: partial names to search to remove those columns
    - colY: name of target
    - binary: boolean detemines whether features will be binary only 
    - shifted: boolean detemines whether rows will be shifted by one

    """
    
    if shiftFeatures==True and shiftTarget==True:
        raise ValueError("Features and Target cannot be shifted at the same time")
        
    #dataset = remove_columns_from_dataset(dataset, predicting = predicting, shifted = shiftFeatures)
    #dataset = dataset[['GOLD_RD_B1_USD_AM','SILVER_RD_B1_USD','PLAT_RD_B1_USD_AM','OIL_BRENT_RD_B1_USD','DJIA_RD_B1_Open','HSI_RD_B1_Open','IBEX_RD_B1_Open','IBEX_RD_B1_Close','N225_RD_B1_Open','SP500_RD_B1_Open']]
    ###########################
    # predicting close price: #
    ###########################    
    colY = ''
    colsToRemove = []

    if predicting == 'close': 
        colY = 'IBEX_RD_B1_Close'
        colsToRemove = ['Date', 'IBEX_RD_B1_Close']
    if predicting == 'open': 
        colY = 'IBEX_RD_B1_Open'
        colsToRemove = ['Date', 'IBEX_RD_B1_Close', 'IBEX_RD_B1_Open']

    train_df = dataset.iloc[trainDates[0]:trainDates[1]+1,]    
    test_df = dataset.iloc[testDates[0]:testDates[1]+1,]
        
    if binary:        
        colsToRemove.extend([col for col in dataset.columns if '_B' not in col])
        colsToRemove = list(set(colsToRemove))
        trainX = np.nan_to_num(np.asarray(train_df.drop(colsToRemove, axis = 1)))
        testX = np.nan_to_num(np.asarray(test_df.drop(colsToRemove, axis = 1)))        
    else:        
        colsToRemove = list(set(colsToRemove))
        trainX = np.nan_to_num(np.asarray(train_df.drop(colsToRemove, axis = 1)))
        testX = np.nan_to_num(np.asarray(test_df.drop(colsToRemove, axis = 1)))

    if shiftTarget:        
        trainY = np.nan_to_num(np.asarray(train_df[colY].shift(1)))[:]
        testY = np.nan_to_num(np.asarray(test_df[colY].shift(1)))[:]
        trainX = trainX[1:]
        testX = testX[1:]
    else:
        if shiftFeatures:
            trainY = np.nan_to_num(np.asarray(train_df[colY].shift(-1)))
            testY = np.nan_to_num(np.asarray(test_df[colY].shift(-1)))
            trainX = trainX[1:-1,1:-1]
            trainY = trainY[1:-1]
            testX = testX[1:-1,1:-1]
            testY = testY[1:-1]
        else:
            trainY = np.nan_to_num(np.asarray(train_df[colY]))
            testY = np.nan_to_num(np.asarray(test_df[colY]))

    #df = df.drop(dataset.index[-1,], axis=0)

    columns_names = dataset.drop(colsToRemove, axis=1).columns.values

    return trainX, trainY, testX, testY, columns_names

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



def train_arrays_experiments(df_x, df_y, trainDates, testDates):
    """
    
    Parameter
    ---------------------
    - dataset: dataframe containing all available columns for a set of dates
    - trainDates: list containing the start training day and end training day
    - testDates: list containing the start training day and end testing day

    """
    train_df_x = df_x.iloc[trainDates[0]:trainDates[1]+1,]    
    train_df_y = df_y.iloc[trainDates[0]:trainDates[1]+1,]    
    test_df_x  = df_x.iloc[testDates[0]:testDates[1]+1,]    
    test_df_y  = df_y.iloc[testDates[0]:testDates[1]+1,]


    trainX = np.nan_to_num(np.asarray(train_df_x))
    testX = np.nan_to_num(np.asarray(test_df_x))
    
    trainY = np.nan_to_num(np.asarray(train_df_y))
    testY = np.nan_to_num(np.asarray(test_df_y))
    
    return trainX, trainY, testX, testY



def only_train_array(df_x, df_y, trainDates):
    """
    
    Parameter
    ---------------------
    - dataset: dataframe containing all available columns for a set of dates
    - trainDates: list containing the start training day and end training day
    - testDates: list containing the start training day and end testing day

    """
    train_df_x = df_x.iloc[trainDates[0]:trainDates[1],]    
    train_df_y = df_y.iloc[trainDates[0]:trainDates[1],]      

    trainX = np.nan_to_num(np.asarray(train_df_x))
    trainY = np.nan_to_num(np.asarray(train_df_y))    
    
    return trainX, trainY

def only_test_array(df_x, df_y, testDates):
    """
    
    Parameter
    ---------------------
    - dataset: dataframe containing all available columns for a set of dates
    - trainDates: list containing the start training day and end training day
    - testDates: list containing the start training day and end testing day

    """

    test_df_x  = df_x.iloc[testDates[0]:testDates[1],]    
    test_df_y  = df_y.iloc[testDates[0]:testDates[1],]

    testX = np.nan_to_num(np.asarray(test_df_x))
    testY = np.nan_to_num(np.asarray(test_df_y))
    
    return testX, testY
