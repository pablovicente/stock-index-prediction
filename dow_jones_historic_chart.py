%matplotlib inline

import six
import Quandl
import calendar
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

historic = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Dow_Jones_Industrial_Average_Trading_Volume_(Recent_History).csv')
historic = historic.iloc[735:]
sb.set_style('darkgrid')

date = [datetime.strptime(d,'%m/%d/%Y').date() for d in historic['Date'].tolist()]
dates = []
values = []
for year in range(1990, 2010, 1):
    cols = [col for col in historic['Date'] if str(year) in col]
    temp = 0
    for col in cols:
        #print historic[historic['Date'] == col].iloc[0]['Value']
        temp = temp + historic[historic['Date'] == col].iloc[0]['Value']
    temp = temp / len(cols)
    #print temp
    dates.append(cols[0])
    values.append(temp)

dict_values = {'Date':dates, 'Values': values}
df_dataset = pd.DataFrame(dict_values)
#df_dataset[cols] = df_dataset[cols].shift(1)
#historic.reset_index(inplace=True)
#historic['Date'] = pd.to_datetime(historic['Date'])
#historic = historic.set_index('Date')

fig_size = [14, 8]
plt.rcParams["figure.figsize"] = fig_size

fig = pylab.figure(0)
#g = sb.factorplot(x="Date", y="Values", data=df_dataset, size=6, kind="bar", palette="muted")
sb.barplot(x="Date", y="Values", data=df_dataset, palette="Greens")

#fig.savefig('/Users/Pablo/Desktop/figure.png')

#sb.set(style="whitegrid")

# Load the example Titanic dataset
#titanic = sb.load_dataset("titanic")

# Draw a nested barplot to show survival for class and sex
#g = sb.factorplot(x="class", y="survived", hue="sex", data=titanic, size=6, kind="bar", palette="muted")