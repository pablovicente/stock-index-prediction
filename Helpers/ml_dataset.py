#!/usr/bin/python

import calendar
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.tsa.stattools as stats
import statsmodels.tsa.stattools as stats


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

    # The Open price of HSI is shifted due to it start way earlier than Madrid
    # All values corresponing to N255 are shifted by 1 due to it closes before Madrid opens
    #cols = [col for col in df_dataset.columns if 'HSI' in col and 'Open' in col]
    #df_dataset[cols] = df_dataset[cols].shift(1)
    #cols = [col for col in df_dataset.columns if 'N225' in col]
    #df_dataset[cols] = df_dataset[cols].shift(1)

    #Sort rows 
    df_dataset = df_dataset[temp_names_list]

    return df_dataset




def dataset_to_train(train_df, test_df, namesToRemove = ['Date', 'IBEX'], colY = 'IBEX_RD_B1_Close', binary = False, shifted = True):
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
    

    colsToRemove = []

    for name in namesToRemove:
        colsToRemove.extend([col for col in train_df.columns if name in col])     

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

def dataset_to_train_using_dates(dataset, trainDates, testDates, namesToRemove = ['Date', 'IBEX'], colY = 'IBEX_RD_B1_Close', binary = False, shifted = True):
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
    colsToRemove = []

    for name in namesToRemove:
        colsToRemove.extend([col for col in dataset.columns if name in col])        

    train_df = dataset.iloc[trainDates[0]:trainDates[1],]
    test_df = dataset.iloc[testDates[0]:testDates[1],]
    
    if binary:
        colsToRemove.extend([col for col in dataset.columns if '_B' not in col])
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

def feature_importance(trainX, trainY, testX, testY):   
    """
    Calculates the feature importance on the training set for a given set of variables
    It prints this importance and plots it 
    """

    ## Feature selection
    clf = ExtraTreesClassifier(random_state=1729, n_estimators=250, n_jobs=-1)
    selector = clf.fit(trainX, trainY)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(trainX.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(trainX.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(trainX.shape[1]), indices)
    plt.xlim([-1, trainX.shape[1]])
    plt.show()


def feature_selection(trainX, trainY, testX, testY):   
    """
    Calculate the feature importance and select the most importance features
    It return the filtered training and testing sets
    """
    ## Feature selection
    clf = ExtraTreesClassifier(random_state=1729, n_estimators=250, n_jobs=-1)
    selector = clf.fit(trainX, trainY)

    fs = SelectFromModel(selector, prefit=True)
    trainX = fs.transform(trainX)
    testX = fs.transform(testX)

    return trainX, testX
