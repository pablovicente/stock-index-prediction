#!/usr/bin/python

"""
It performs an analysis of an stock index. It plots its value, day price difference and variance. 
It plots a scatterplot of the price at time t and lagged time. 
It also seeks for seasonal periods within the time series

"""

stock_data = INDEX_SP500
#date = [datetime.strptime(d,'%Y-%m-%d').date() for d in stock_data['Date'].tolist()]

#stock_data.reset_index(inplace=True)
#stock_data['Date'] = pd.to_datetime(stock_data['Date'])
#stock_data = stock_data.set_index('Date')

#stock_data['Close'].plot(figsize=(16, 12))  

stock_data['First Difference'] = stock_data['Close'] - stock_data['Close'].shift()  
#stock_data['First Difference'].plot(figsize=(16, 12))  

stock_data['Natural Log'] = stock_data['Close'].apply(lambda x: np.log(x))  
#stock_data['Natural Log'].plot(figsize=(16, 12))  

stock_data['Original Variance'] = pd.rolling_var(stock_data['Close'], 30, min_periods=None, freq=None, center=True)  
stock_data['Log Variance'] = pd.rolling_var(stock_data['Natural Log'], 30, min_periods=None, freq=None, center=True)
stock_data['Log Variance'] = pd.rolling_var(stock_data['Natural Log'], 30, min_periods=None, freq=None, center=True)

fig, ax = plt.subplots(2, 1, figsize=(13, 12))  
#stock_data['Original Variance'].plot(ax=ax[0], title='Original Variance')  
#stock_data['Log Variance'].plot(ax=ax[1], title='Log Variance')  
#fig.tight_layout()  

stock_data['Logged First Difference'] = stock_data['Natural Log'] - stock_data['Natural Log'].shift()  
#stock_data['Logged First Difference'].plot(figsize=(16, 12))  

stock_data['Lag 1'] = stock_data['Logged First Difference'].shift()  
stock_data['Lag 2'] = stock_data['Logged First Difference'].shift(2)  
stock_data['Lag 5'] = stock_data['Logged First Difference'].shift(5)  
stock_data['Lag 30'] = stock_data['Logged First Difference'].shift(30)

#sb.jointplot('Logged First Difference', 'Lag 1', stock_data, kind='reg', size=13)  

lag_correlations = stats.acf(stock_data['Natural Log'].iloc[1:])  
lag_partial_correlations = stats.pacf(stock_data['Logged First Difference'].iloc[1:])  

fig, ax = plt.subplots(figsize=(16,12))  
ax.plot(lag_correlations, marker='o', linestyle='--') 

#decomposition = seasonal.seasonal_decompose(stock_data['Natural Log'], model='additive', freq=30)  
#fig = plt.figure()  
#fig = decomposition.plot()  
#
#model = sm.tsa.ARIMA(stock_data['Natural Log'].iloc[1:], order=(1, 0, 0))  
#results = model.fit(disp=-1)  
#stock_data['Forecast'] = results.fittedvalues  
#stock_data[['Natural Log', 'Forecast']].plot(figsize=(16, 12))  
#
#model = sm.tsa.ARIMA(stock_data['Logged First Difference'].iloc[1:], order=(1, 0, 0))  
#results = model.fit(disp=-1)  
#stock_data['Forecast'] = results.fittedvalues  
#stock_data[['Logged First Difference', 'Forecast']].plot(figsize=(16, 12))  
#
#stock_data[['Logged First Difference', 'Forecast']].iloc[1200:1600, :].plot(figsize=(16, 12))  
#
#model = sm.tsa.ARIMA(stock_data['Logged First Difference'].iloc[1:], order=(0, 0, 1))  
#results = model.fit(disp=-1)  
#stock_data['Forecast'] = results.fittedvalues  
#stock_data[['Logged First Difference', 'Forecast']].plot(figsize=(16, 12))