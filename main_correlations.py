#!/usr/bin/python

"""
It calculates the autocorrelation of the diffent indices and plots it.
It also calculates the cross correlation between indices and plots it.

In order to obtain the real correlation it it necessary to use the day price difference.
Due to the huge range value of the stocks value, the correlation provides us a false result. 
"""

#####################
#  AUTOCORRELATION  #
#####################

x = [datetime.strptime(d,'%Y-%m-%d').date() for d in INDEX_DJIA['Date'].tolist()]

x_array = np.asarray(x)
nlag = len(x)
range_show = nlag

acf_gold = correlation.autocorrelation(GOLD['USD (AM)'], nlag, True)
acf_silver = correlation.autocorrelation(SILVER['USD'], nlag, True)
acf_plat = correlation.autocorrelation(PLAT['USD AM'], nlag, True)

acf_djia = correlation.autocorrelation(INDEX_DJIA['Close'], nlag, True)
acf_hsi = correlation.autocorrelation(INDEX_HSI['Close'], nlag, True)
acf_ibex = correlation.autocorrelation(INDEX_IBEX['Close'], nlag, True)
acf_n225 = correlation.autocorrelation(INDEX_N225['Close'], nlag, True)
acf_sp500 = correlation.autocorrelation(INDEX_SP500['Close'], nlag, True)


plt.rcParams["figure.figsize"] = fig_size
pylab.figure(0)

ax = plt.subplot(111)
ax.plot(x, acf_gold, 'b', label='Gold')
ax.plot(x, acf_silver, 'r', label='Silver')
ax.plot(x, acf_plat, 'g', label='Plat')


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

fig = pylab.figure(1)

ax = plt.subplot(111)

ax.plot(x_array[:range_show,], acf_djia[:range_show,], 'b', label='DJIA')
ax.plot(x_array[:range_show,], acf_hsi[:range_show,], 'r', label='HSI')
ax.plot(x_array[:range_show,], acf_ibex[:range_show,], 'g', label='IBEX')
ax.plot(x_array[:range_show,], acf_n225[:range_show,], 'y', label='N225')
ax.plot(x_array[:range_show,], acf_sp500[:range_show,], 'k', label='SP500')
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#fig.savefig('/Users/Pablo/Desktop/figure.png')


######################
#  CROSSCORRELATION  #
######################

colors_ = list(six.iteritems(colors.cnames))
x = [datetime.strptime(d,'%Y-%m-%d').date() for d in INDEX_DJIA['Date'].tolist()]
fig = pylab.figure(0)
ax = plt.subplot(111)

cross_correlations = []
plots = []
color_iter = iter(colors_)
for i in range(1):
    for j in range(len(values)):
        if i != j:            
            temp = correlation.crosscorrelation(values[i], values[j], True)
            legend = str(values_names[i]) + ' ' + str(values_names[j])
            cross_correlations.append(temp)  
            color = next(color_iter)[0]
            plots.append(ax.plot( temp, color, label=legend))
            next(color_iter)
            print("Cross correlation between %s and %s, color %s" % (values_names[i], values_names[j], color))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

#fig.savefig('/Users/Pablo/Desktop/figure.png')

#########################
#  CROSSCORRELATION  2  #
#########################

sb.set_style('darkgrid')

date = [datetime.strptime(d,'%Y-%m-%d').date() for d in INDEX_IBEX['Date'].tolist()]

INDEX_IBEX.reset_index(inplace=True)
INDEX_IBEX['Date'] = pd.to_datetime(INDEX_IBEX['Date'])
INDEX_IBEX = INDEX_IBEX.set_index('Date')

INDEX_DJIA.reset_index(inplace=True)
INDEX_DJIA['Date'] = pd.to_datetime(INDEX_DJIA['Date'])
INDEX_DJIA = INDEX_DJIA.set_index('Date')

INDEX_HSI.reset_index(inplace=True)
INDEX_HSI['Date'] = pd.to_datetime(INDEX_HSI['Date'])
INDEX_HSI = INDEX_HSI.set_index('Date')

INDEX_N225.reset_index(inplace=True)
INDEX_N225['Date'] = pd.to_datetime(INDEX_N225['Date'])
INDEX_N225 = INDEX_N225.set_index('Date')

INDEX_SP500.reset_index(inplace=True)
INDEX_SP500['Date'] = pd.to_datetime(INDEX_SP500['Date'])
INDEX_SP500 = INDEX_SP500.set_index('Date')



col_iter = iter(values_cols)
for df in values_dfs:    
    col = next(col_iter) 
    df['First Difference'] = df[col] - df[col].shift()
    df['Natural Log'] = df[col].apply(lambda x: np.log(x))  
    df['Logged First Difference'] = df['Natural Log'] - df['Natural Log'].shift()  
    df['Lag 1'] = df['First Difference'].shift()  
    df['Lag 2'] = df['First Difference'].shift(2)  
    df['Lag 5'] = df['First Difference'].shift(5)  
    df['Lag 30'] = df['First Difference'].shift(30)  
    df['Logged Lag 1'] = df['Logged First Difference'].shift()  
    df['Logged Lag 2'] = df['Logged First Difference'].shift(2)  
    df['Logged Lag 5'] = df['Logged First Difference'].shift(5)  
    df['Logged Lag 30'] = df['Logged First Difference'].shift(30)  




#GOLD['First Difference'] = GOLD['USD_AM'] - GOLD['USD_AM'].shift()  
#GOLD['Natural Log'] = GOLD['USD_AM'].apply(lambda x: np.log(x)) 
#GOLD['Logged First Difference'] = GOLD['Natural Log'] - GOLD['Natural Log'].shift()  
nlags = 40
lag_correlations_IBEX_GOLD      = tsa_stattools.ccf(INDEX_IBEX['Lag 1'].iloc[2:],        GOLD['Lag 1'].iloc[2:], unbiased=True)
lag_correlations_IBEX_SILVER    = tsa_stattools.ccf(INDEX_IBEX['Lag 1'].iloc[2:],      SILVER['Lag 1'].iloc[2:],    unbiased=True)
lag_correlations_IBEX_PLAT      = tsa_stattools.ccf(INDEX_IBEX['Lag 1'].iloc[2:],        PLAT['Lag 1'].iloc[2:], unbiased=True)
lag_correlations_IBEX_OIL_BRENT = tsa_stattools.ccf(INDEX_IBEX['Lag 1'].iloc[2:],   OIL_BRENT['Lag 1'].iloc[2:],    unbiased=True)
lag_correlations_IBEX_DJIA      = tsa_stattools.ccf(INDEX_IBEX['Lag 1'].iloc[2:],  INDEX_DJIA['Lag 1'].iloc[2:],   unbiased=True)
lag_correlations_IBEX_HSI       = tsa_stattools.ccf(INDEX_IBEX['Lag 1'].iloc[2:],   INDEX_HSI['Lag 1'].iloc[2:],   unbiased=True)
lag_correlations_IBEX_IBEX      = tsa_stattools.ccf(INDEX_IBEX['Lag 1'].iloc[2:],  INDEX_IBEX['Lag 1'].iloc[2:],   unbiased=True)
lag_correlations_IBEX_N225      = tsa_stattools.ccf(INDEX_IBEX['Lag 1'].iloc[2:],  INDEX_N225['Lag 1'].iloc[2:],   unbiased=True)
lag_correlations_IBEX_SP500     = tsa_stattools.ccf(INDEX_IBEX['Lag 1'].iloc[2:], INDEX_SP500['Lag 1'].iloc[2:],   unbiased=True)




fig_size = [12, 8]
plt.rcParams["figure.figsize"] = fig_size

pylab.figure(1)
ax = plt.subplot(111)

colors_ = list(six.iteritems(colors.cnames))
x = [datetime.strptime(d,'%Y-%m-%d').date() for d in INDEX_DJIA['Date'].tolist()]
fig = pylab.figure(0)
ax = plt.subplot(111)
color_iter = iter(colors_)
nlag = len(x)
print len(x)
print (len(lag_correlations_IBEX_DJIA))
ax.plot(lag_correlations_IBEX_GOLD[0:nlag,], label='IBEX-GOLD', color='gold')
ax.plot(lag_correlations_IBEX_SILVER[0:nlag,], label='IBEX-SILVER')
ax.plot(lag_correlations_IBEX_PLAT[0:nlag,], label='IBEX-PLAT', color='lightgreen')
ax.plot(lag_correlations_IBEX_OIL_BRENT[0:nlag,], label='IBEX-OIL BRENT', color='yellow')
ax.plot(lag_correlations_IBEX_DJIA[0:nlag,], label='IBEX-DJIA')
ax.plot(lag_correlations_IBEX_HSI[0:nlag,], label='IBEX-HSI', color='aqua')
ax.plot(lag_correlations_IBEX_IBEX[0:nlag,], label='IBEX-IBEX', color='black')
ax.plot(lag_correlations_IBEX_N225[0:nlag,], label='IBEX-N225', color='deeppink')
ax.plot(lag_correlations_IBEX_SP500[0:nlag,], label='IBEX-SP500', color='darkkhaki')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Fecha')
plt.ylabel('Correlacion')

plt.savefig('Images/crosscorrelation_lag_1_biased.png')