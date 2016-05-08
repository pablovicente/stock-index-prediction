#!/usr/bin/python

####################
#   STOCK INDICES  #
####################

INDEX_DJIA = procces_stocks.order_dataframe(INDEX_DJIA)
#INDEX_DJIA no need to multiply it
INDEX_HSI = procces_stocks.order_dataframe(INDEX_HSI)
#INDEX_HSI no need to multiply it
INDEX_IBEX = procces_stocks.order_dataframe(INDEX_IBEX)
#INDEX_IBEX no need to multiply it
INDEX_N225 = procces_stocks.order_dataframe(INDEX_N225)
#INDEX_N225 no need to multiply it
INDEX_SP500= procces_stocks.order_dataframe(INDEX_SP500)
#INDEX_SP500 no need to multiply it


min_date = '1993-04-12'
INDEX_DJIA_new = procces_stocks.select_rows_by_actual_date(INDEX_DJIA, min_date)
INDEX_HSI_new = procces_stocks.select_rows_by_actual_date(INDEX_HSI, min_date)
INDEX_IBEX_new = procces_stocks.select_rows_by_actual_date(INDEX_IBEX, min_date)
INDEX_N225_new = procces_stocks.select_rows_by_actual_date(INDEX_N225, min_date)
INDEX_SP500_new = procces_stocks.select_rows_by_actual_date(INDEX_SP500, min_date)

cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted Close']
INDEX_HSI_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, INDEX_HSI_new)
INDEX_IBEX_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, INDEX_IBEX_new)
INDEX_N225_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, INDEX_N225_new)

cols = ['Open','High','Low','Close','Volume','Adjusted Close']
INDEX_HSI_new = fill_gaps_with_interpolation(INDEX_HSI, INDEX_HSI_new, cols, ['linear'], False)
INDEX_IBEX_new = fill_gaps_with_interpolation(INDEX_IBEX, INDEX_IBEX_new, cols, ['linear'], False)
INDEX_N225_new = fill_gaps_with_interpolation(INDEX_N225, INDEX_N225_new, cols, ['linear'], False)

cols = ['Volume']
INDEX_HSI_new = fill_gaps_with_interpolation(INDEX_HSI, INDEX_HSI_new, cols, ['linear'], False, 0.0)
INDEX_IBEX_new = fill_gaps_with_interpolation(INDEX_IBEX, INDEX_IBEX_new, cols, ['linear'], False, 0.0)
INDEX_N225_new = fill_gaps_with_interpolation(INDEX_N225, INDEX_N225_new, cols, ['linear'], False, 0.0)

print has_gaps(INDEX_DJIA_new, cols, 0.0)
print has_gaps(INDEX_SP500_new, cols, 0.0)
print has_gaps(INDEX_HSI_new, cols, 0.0)
print has_gaps(INDEX_IBEX_new, cols, 0.0)
print has_gaps(INDEX_N225_new, cols, 0.0)

# Set figure width to 12 and height to 9
fig_size = [10, 6]
plt.rcParams["figure.figsize"] = fig_size
plt.plot(INDEX_SP500_new['Close'], 'b')
plt.plot(INDEX_IBEX_new['Close'], 'g')
plt.plot(INDEX_DJIA_new['Close'], 'r')
plt.plot(INDEX_HSI_new['Open'], 'y')
plt.plot(INDEX_N225_new['Open'], 'c')
plt.show()


####################
#    COMMODITIES   #
####################


GOLD = procces_stocks.order_dataframe(GOLD)
SILVER = procces_stocks.order_dataframe(SILVER)
PLAT = procces_stocks.order_dataframe(PLAT)

min_date = '1993-04-08'
GOLD_new = procces_stocks.select_rows_by_actual_date(GOLD, min_date)
SILVER_new = procces_stocks.select_rows_by_actual_date(SILVER, min_date)
PLAT_new = procces_stocks.select_rows_by_actual_date(PLAT, min_date)

cols = ['Date', 'USD (AM)', 'USD (PM)']
GOLD_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA, GOLD_new, cols)
cols = ['Date', 'USD']
SILVER_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA, SILVER_new, cols)
cols = ['Date', 'USD AM', 'USD PM']
PLAT_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA, PLAT_new, cols)

cols = ['USD (AM)', 'USD (PM)']
GOLD_new = procces_stocks.fill_gaps_with_interpolation(GOLD, GOLD_new, cols, ['linear'], False)
cols = ['USD']
SILVER_new = procces_stocks.fill_gaps_with_interpolation(SILVER, SILVER_new, cols, ['linear'], False)
cols = ['USD AM', 'USD PM']
PLAT_new = procces_stocks.fill_gaps_with_interpolation(PLAT, PLAT_new, cols, ['linear'], False)

fig_size = [10, 6]
plt.rcParams["figure.figsize"] = fig_size
plt.plot(GOLD_new['USD (AM)'], 'b')
plt.plot(SILVER_new['USD'], 'g')
plt.plot(PLAT_new['USD AM'], 'r')
plt.show()


#####################
#  AUTOCORRELATION  #
#####################

x = [datetime.strptime(d,'%Y-%m-%d').date() for d in INDEX_DJIA['Date'].tolist()]
nlag = len(x)

acf_gold = correlation.autocorrelationf(GOLD['USD (AM)'], nlag, True)
acf_silver = correlation.autocorrelationf(SILVER['USD'], nlag, True)
acf_plat = correlation.autocorrelationf(PLAT['USD AM'], nlag, True)

acf_djia = correlation.autocorrelationf(INDEX_DJIA['Close'], nlag, True)
acf_hsi = correlation.autocorrelationf(INDEX_HSI['Close'], nlag, True)
acf_ibex = correlation.autocorrelationf(INDEX_IBEX['Close'], nlag, True)
acf_n225 = correlation.autocorrelationf(INDEX_N225['Close'], nlag, True)
acf_sp500 = correlation.autocorrelationf(INDEX_SP500['Close'], nlag, True)
                
ccf = correlation.crosscorrelation(GOLD['USD (AM)'], GOLD['USD (AM)'], False)
plt.rcParams["figure.figsize"] = fig_size
pylab.figure(0)

ax = plt.subplot(111)
ax.plot(acf_gold, 'b', label='Gold')
ax.plot(acf_silver, 'r', label='Silver')
ax.plot(acf_plat, 'g', label='Plat')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

pylab.figure(1)

ax = plt.subplot(111)
ax.plot(acf_djia, 'b', label='DJIA')
ax.plot(acf_hsi, 'r', label='HSI')
ax.plot(acf_ibex, 'g', label='IBEX')
ax.plot(acf_n225, 'y', label='N225')
ax.plot(acf_sp500, 'k', label='SP500')
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

######################
#    SAVING DATA     #
######################

data_manipulation.write_csv_data(GOLD, '/Users/Pablo/Desktop/TFM/Data/GOLD.csv')
data_manipulation.write_csv_data(SILVER, '/Users/Pablo/Desktop/TFM/Data/SILVER.csv')
data_manipulation.write_csv_data(PLAT, '/Users/Pablo/Desktop/TFM/Data/PLAT.csv')
data_manipulation.write_csv_data(INDEX_DJIA, '/Users/Pablo/Desktop/TFM/Data/INDEX_DJIA.csv')
data_manipulation.write_csv_data(INDEX_HSI, '/Users/Pablo/Desktop/TFM/Data/INDEX_HSI.csv')
data_manipulation.write_csv_data(INDEX_IBEX, '/Users/Pablo/Desktop/TFM/Data/INDEX_IBEX.csv')
data_manipulation.write_csv_data(INDEX_N225, '/Users/Pablo/Desktop/TFM/Data/INDEX_N225.csv')
data_manipulation.write_csv_data(INDEX_SP500, '/Users/Pablo/Desktop/TFM/Data/INDEX_SP500.csv')

######################
#  CROSSCORRELATION  #
######################

colors_ = list(six.iteritems(colors.cnames))
x = [datetime.strptime(d,'%Y-%m-%d').date() for d in INDEX_DJIA['Date'].tolist()]
pylab.figure(0)
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
