"""
data_manipulation.py

Original author: Pablo Vicente 
"""


import pandas as pd

def read_csv_data(filename):
    """
        Read a csv file from disk
        """
    return pd.read_csv(filename)

def write_csv_data(df, filename, row_names=False, col_names=True):
    """
        Write a csv file to disk
        """
    df.to_csv(filename, index=row_names, header=col_names)