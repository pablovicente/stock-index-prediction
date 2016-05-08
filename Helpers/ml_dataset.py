#!/usr/bin/python

import calendar
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.tsa.stattools as stats
import statsmodels.tsa.stattools as stats

def generate_df_dataset(names_list, df_list, cols_list):
    """
    
    """
    if len(names_list) != len(df_list) or len(names_list) != len(cols_list):
        print('The three list must have the same length')
        exit(0)        
    
    rows =  df_list[0].shape[0]
    cols =  len(df_list)
    dates = df_list[0]['Date'].tolist()

    dict_values = {'Date':dates}
    for index in range(len(names_list)):
        name = names_list[index]
        df = df_list[index]
        col = cols_list[index]     

        dict_values[name] = df[col]

    temp_names_list = list(names_list)
    temp_names_list.insert(0, 'Date')
    df_dataset = pd.DataFrame(dict_values)
    df_dataset = df_dataset[temp_names_list]

    return df_dataset