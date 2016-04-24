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