#!/usr/bin/python

import math
import calendar
import numpy as np
import pandas as pd

from datetime import datetime
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stats
import statsmodels.tsa.stattools as stats


def order_dataframe(df, col='Date', ascending=True):
    """
    """
    df = df.sort_values(col, ascending=ascending)
    df.index = range(df.shape[0])
    return df

def multiply_df(df, value, cols):
    """
    """
    df[cols] = df[cols].apply(lambda x: x*value)
    return df

def select_rows_by_actual_date(df, min_date):
    """
    """
    rows =  df.Date[df.Date == min_date].index[0]
    rows = df.shape[0] - rows - 1
    df_new = df.tail(rows+1) 
    return order_dataframe(df_new)


def compare_dates(df1, df2):
    """
    """
    equal = True
    for i in  df1.index[::1]:
        if df1.Date[i] != df2.Date[i]:
            print("1 - Different dates at", i)
            print(df1.Date[i], df2.Date[i])
            equal = False

    return equal

def get_index_of_different_dates(df1, df2):
    """
    """
    values = df1.Date.isin(df2.Date)
    for i in range(len(values)): 
        if values[i] == False:
            indices =  df1.Date[i].index()
            
    return values, indices

def align_date_in_dataframe(df1, df2, cols):
    """
    Align second dataframe respect to one using Date column
    """
    
    if check_dataframes_alignment(df1, df2) == True:
        print "Dataframes already aligned"
        return df2     
    
    values = df2.Date.isin(df1.Date)
    for i in range(len(values)): 
        if values[i] == False:
            df2 = df2[df2.Date != df2.Date[i]]
            
    dict_df = {}
    for i in range(len(cols)):
        dict_df[cols[i]] = -1
        
    values = df1.Date.isin(df2.Date)        
    for i in range(len(values)): 
        if values[i] == False:
            dict_df['Date'] = df1.Date[i]
            line = pd.DataFrame(dict_df, index=[i])    
            df2 = df2.append(line, ignore_index=True)

    df2 = df2[cols]
            
    df2 = order_dataframe(df2)
    df2.index = range(df2.shape[0])
    df1.index = range(df1.shape[0])
    return df2

def check_dataframes_alignment(df1, df2):
    """
    Checks whether two dataframes are aligned by date
    """
    values1 = df1.Date.isin(df2.Date)
    values2 = df2.Date.isin(df1.Date)
    if (values1[values1 == False].shape[0] > 0) or (values2[values2 == False].shape[0]):
        return False
    else:
        return True

    
def toTimestamp(d):
    """
    """
    return calendar.timegm(d.timetuple())

def fill_gaps_with_interpolation(df_old, df_new, cols, kind=['linear'], chart=False, gap_value=-1, verbose=True):
    """
    Undertakes interpolation to fill up missing values
    """
    for col in cols:
        if verbose: print("Interpolating column %s..." % (col))
            
        x0 = []
        for i in range(df_old.shape[0]):
            date_object = datetime.strptime(df_old.Date[i], '%Y-%m-%d')
            x0.append(toTimestamp(date_object))

        y0 = df_old.loc[:,(col)].as_matrix() #np.random.random(21)
 
        if(chart == True):
            plt.plot(x0, y0, 'o', label='Data')

        # Array with points in between those of the data set for interpolation.
        x = []
        for i in range(df_new.shape[0]):
            date_object = datetime.strptime(df_new.Date[i], '%Y-%m-%d')
            x.append(toTimestamp(date_object))

        # Available options for interp1d ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 10)
        f = interp1d(x0, y0, kind='linear')    # interpolation function
        
        indices = df_new[col][df_new[col] == gap_value]        
        for i in indices.index:
            date = df_new.Date[i]
            df_new.loc[:,(col)].iloc[i] = f(toTimestamp(datetime.strptime(date, '%Y-%m-%d')))  

        if(chart == True):
            plt.plot(x, f(x), label='linear')      # plot of interpolated data
            plt.legend()
            plt.show()

    return df_new

def has_gaps(df, cols, gap_value=-1):
    """
    """
    for col in cols:
        gaps = df.loc[:,(col)][df.loc[:,(col)] == gap_value].values.shape[0]
        if gaps > 0:
            return True

    return False    

def difference_between_consecutive_days(df, colnames, shift_values):
    """
    Calculates the difference between n day and n+1 day
    If price is higher it set the n+1 column value to 1, otherwise to 0  
    
    It return the number of days that the value increases and the number of days the value decreases
    """
    for shift_value in shift_values:
        for colname in colnames:
            num = str(shift_value) + "_" + colname
            df.loc[:,('RD'+num)] = df[colname] - df[colname].shift(shift_value)    
            df.loc[:,('RD_P'+num)] = ((df[colname] - df[colname].shift(shift_value)) / df[colname].shift(shift_value)) * 100
            df.loc[:,('RD_B'+num)] = df.loc[:,('RD'+num)].apply(lambda x: 1 if x > 0 else 0)
    
    #ones = GOLD.loc[:,('DIFF')][df.loc[:,('DIFF')] == 1].values.shape[0]
    #zeros = GOLD.loc[:,('DIFF')][df.loc[:,('DIFF')] == 0].values.shape[0]    
    return df


def log_return(df, colnames, shift_values):
    """
        
    """
    for shift_value in shift_values:
        for colname in colnames:
            num = "Log_Return_" + str(shift_value) + "_" + colname
            df.loc[:,(num)] = df[colname]/df[colname].shift(shift_value)    
            df.loc[:,(num)] = df.loc[:,(num)].apply(lambda x: math.log(x))    
    
    return df

def consecutive_days_tendency(df, colname):
    """
    """
    max_zeros = temp_zeros = 0
    max_ones = temp_ones = 0
    
    for i in range(df.shape[0]):
        if df[colname][i] == -1:
            temp_zeros = temp_zeros + 1  
            if temp_ones > max_ones:
                max_ones = temp_ones
                date_ones = df['Date'][i-max_ones]
            temp_ones = 0
        else:
            temp_ones = temp_ones + 1    
            if temp_zeros > max_zeros:
                max_zeros = temp_zeros
                date_zeros = df['Date'][i-max_zeros]
            temp_zeros = 0

    return max_zeros, max_ones, date_zeros, date_ones
