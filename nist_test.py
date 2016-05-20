%matplotlib inline

import six
import Quandl
import calendar
import math
import numpy as np
import pandas as pd
import seaborn as sb
import pylab as pylab
from datetime import datetime
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
import statsmodels.stats.stattools as stats_stattools
import statsmodels.tsa.stattools as tsa_stattools
import statsmodels.tsa.seasonal as tsa_seasonal
import statsmodels.api as sm
import xgboost as xgb
from unbalanced_dataset import SMOTE

from sklearn import svm
from sklearn import metrics, cross_validation, linear_model, ensemble
from sklearn import feature_selection
from sklearn import decomposition

import sys
from os import listdir
from os.path import isfile, join
from helpers import correlation, procces_stocks, data_manipulation, download_quandl_data, ml_dataset
from classes import Iteration, Stacking, Boosting

fig_size = [10, 6]
plt.rcParams["figure.figsize"] = fig_size
sb.set_style('darkgrid')

GOLD = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Data/GOLD.csv')
SILVER = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Data/SILVER.csv')
PLAT = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Data/PLAT.csv')
OIL_BRENT = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Data/OIL_BRENT.csv')
INDEX_DJIA = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Data/INDEX_DJIA.csv')
INDEX_HSI = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Data/INDEX_HSI.csv')
INDEX_IBEX = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Data/INDEX_IBEX.csv')
INDEX_N225 = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Data/INDEX_N225.csv')
INDEX_SP500 = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Data/INDEX_SP500.csv')

##################
#      TEST     #
##################

example_binary_string = ""
for i in range(1, INDEX_IBEX['RD_B1_Open'].shape[0], 1):
    example_binary_string+=str(INDEX_IBEX['RD_B1_Open'].iloc[i])

p_value = rng_tester.monobit(example_binary_string)
p_value = rng_tester.block_frequency(example_binary_string, block_size=64)
p_value = rng_tester.independent_runs(example_binary_string)
p_value = rng_tester.longest_runs(example_binary_string)
p_value = rng_tester.matrix_rank(example_binary_string, matrix_size=16)
p_value = rng_tester.spectral(example_binary_string)
p_value = rng_tester.non_overlapping_patterns(example_binary_string, pattern="000000001", num_blocks=8)
p_value = rng_tester.non_overlapping_patterns(example_binary_string, pattern_size=9, block_size=1032)
p_value = rng_tester.universal(example_binary_string)
p_value = rng_tester.linear_complexity(example_binary_string, block_size=500)
p_value = rng_tester.linear_complexity(example_binary_string, pattern_length=16, method="both”)
p_value = rng_tester.approximate_entropy(example_binary_string, pattern_length=16)
p_value = rng_tester.cumulative_sums(example_binary_string, method="forward”)
p_values = rng_tester.random_excursions(example_binary_string)
p_values = rng_tester.random_excursions_variant(example_binary_string)