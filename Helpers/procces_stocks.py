#!/usr/bin/python

import calendar
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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

def align_date_in_dataframe(df1, df2):
    """
    Align second dataframe respect to one using Date column
    """
    
    if check_dataframes_alignment(df1, df2) == True:
        return     
    
    values = df2.Date.isin(df1.Date)
    for i in range(len(values)): 
        if values[i] == False:
            df2 = df2[df2.Date != df2.Date[i]]

    values = df1.Date.isin(df2.Date)        
    for i in range(len(values)): 
        if values[i] == False:
            line = pd.DataFrame({"Date": df1.Date[i], "Open": -1, "High": -1, "Low": -1, "Close": -1, "Volume": -1, "Adjusted Close": -1}, index=[i])    
            df2 = df2.append(line, ignore_index=True)

    df2 = df2[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted Close']]
            
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
        for i in range(INDEX_IBEX_new.shape[0]):
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