##############################
#         READ FILES         #
##############################

GOLD = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/LBMA-GOLD.csv')
SILVER = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/LBMA-SILVER.csv')
PLAT = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/LPPM-PLAT.csv')
OIL_BRENT = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/FRED-DCOILBRENTEU.csv')

USD_GBP = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/BUNDESBANK-USD_GBP.csv')
JPY_USD = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/JPY_USD.csv')
AUD_USD = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/RBA-AUD_USD.csv')

INDEX_DJIA = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/YAHOO-INDEX_DJIA.csv')
INDEX_HSI = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/YAHOO-INDEX_HSI.csv')
INDEX_IBEX = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/YAHOO-INDEX_IBEX.csv')
INDEX_N225 = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/YAHOO-INDEX_N225.csv')
INDEX_SP500 = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/YAHOO-INDEX_SP500.csv')
INDEX_AXJO = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/YAHOO-INDEX_AXJO.csv')
INDEX_FCHI = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/YAHOO-INDEX_FCHI.csv')
INDEX_GDAXI = data_manipulation.read_csv_data('/Users/Pablo/Desktop/TFM/Raw_Data/YAHOO-INDEX_GDAXI.csv')

##############################
#        DELETE COLS         #
##############################


GOLD = GOLD.drop([col for col in GOLD.columns if 'USD_AM' in col or 'GBP (AM)' in col or 'GBP (PM)' in col or 'EURO (AM)' in col or 'EURO (PM)' in col], axis = 1)
SILVER = SILVER.drop([col for col in SILVER.columns if 'GPB' in col or 'EURO' in col], axis = 1)
PLAT = PLAT.drop([col for col in PLAT.columns if 'USD_AM' in col or  'GBP AM' in col or 'GBP PM' in col or 'EURO AM' in col or 'EURO PM' in col], axis = 1)

INDEX_DJIA = INDEX_DJIA.drop([col for col in INDEX_DJIA.columns if 'Adjusted Close' in col], axis = 1)
INDEX_SP500 = INDEX_SP500.drop([col for col in INDEX_SP500.columns if 'Adjusted Close' in col], axis = 1)
INDEX_IBEX = INDEX_IBEX.drop([col for col in INDEX_IBEX.columns if 'Volume' in col or 'Adjusted Close' in col], axis = 1)
INDEX_HSI = INDEX_HSI.drop([col for col in INDEX_HSI.columns if 'Volume' in col or 'Adjusted Close' in col], axis = 1)
INDEX_N225 = INDEX_N225.drop([col for col in INDEX_N225.columns if 'Volume' in col or 'Adjusted Close' in col], axis = 1)
INDEX_AXJO = INDEX_AXJO.drop([col for col in INDEX_AXJO.columns if 'Volume' in col or 'Adjusted Close' in col], axis = 1)
INDEX_FCHI = INDEX_FCHI.drop([col for col in INDEX_FCHI.columns if 'Volume' in col or 'Adjusted Close' in col], axis = 1)
INDEX_GDAXI = INDEX_GDAXI.drop([col for col in INDEX_GDAXI.columns if 'Volume' in col or 'Adjusted Close' in col], axis = 1)

##############################
#        DELETE COLS         #
##############################

GOLD = procces_stocks.order_dataframe(GOLD)
SILVER = procces_stocks.order_dataframe(SILVER)
PLAT = procces_stocks.order_dataframe(PLAT)
OIL_BRENT = procces_stocks.order_dataframe(OIL_BRENT)

USD_GBP = procces_stocks.order_dataframe(USD_GBP)
JPY_USD = procces_stocks.order_dataframe(JPY_USD)
AUD_USD = procces_stocks.order_dataframe(AUD_USD)

INDEX_DJIA = procces_stocks.order_dataframe(INDEX_DJIA)
INDEX_HSI = procces_stocks.order_dataframe(INDEX_HSI)
INDEX_IBEX = procces_stocks.order_dataframe(INDEX_IBEX)
INDEX_N225 = procces_stocks.order_dataframe(INDEX_N225)
INDEX_SP500= procces_stocks.order_dataframe(INDEX_SP500)
INDEX_AXJO = procces_stocks.order_dataframe(INDEX_AXJO)
INDEX_FCHI = procces_stocks.order_dataframe(INDEX_FCHI)
INDEX_GDAXI= procces_stocks.order_dataframe(INDEX_GDAXI)

##############################
#        DELETE COLS         #
##############################

min_date = '1993-07-07'
GOLD_new = procces_stocks.select_rows_by_actual_date(GOLD, min_date)
SILVER_new = procces_stocks.select_rows_by_actual_date(SILVER, min_date)
PLAT_new = procces_stocks.select_rows_by_actual_date(PLAT, min_date)
OIL_BRENT_new = procces_stocks.select_rows_by_actual_date(OIL_BRENT, min_date)

USD_GBP_new = procces_stocks.select_rows_by_actual_date(USD_GBP, min_date)
JPY_USD_new = procces_stocks.select_rows_by_actual_date(JPY_USD, min_date)
AUD_USD_new = procces_stocks.select_rows_by_actual_date(AUD_USD, min_date)

INDEX_DJIA_new = procces_stocks.select_rows_by_actual_date(INDEX_DJIA, min_date)
INDEX_HSI_new = procces_stocks.select_rows_by_actual_date(INDEX_HSI, min_date)
INDEX_IBEX_new = procces_stocks.select_rows_by_actual_date(INDEX_IBEX, min_date)
INDEX_N225_new = procces_stocks.select_rows_by_actual_date(INDEX_N225, min_date)
INDEX_SP500_new = procces_stocks.select_rows_by_actual_date(INDEX_SP500, min_date)
INDEX_AXJO_new = procces_stocks.select_rows_by_actual_date(INDEX_AXJO, min_date)
INDEX_FCHI_new = procces_stocks.select_rows_by_actual_date(INDEX_FCHI, min_date)
INDEX_GDAXI_new = procces_stocks.select_rows_by_actual_date(INDEX_GDAXI, min_date)

##############################
#        DELETE COLS         #
##############################

cols = ['Date', 'USD']
GOLD_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, GOLD_new, cols)
SILVER_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, SILVER_new, cols)
PLAT_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, PLAT_new, cols)
OIL_BRENT_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, OIL_BRENT_new, cols)

cols = ['Date', 'Value']
USD_GBP_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, USD_GBP_new, cols)
JPY_USD_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, JPY_USD_new, cols)
AUD_USD_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, AUD_USD_new, cols)

cols = ['Date', 'Open', 'High', 'Low', 'Close']
INDEX_HSI_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, INDEX_HSI_new, cols)
INDEX_IBEX_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, INDEX_IBEX_new, cols)
INDEX_N225_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, INDEX_N225_new, cols)
INDEX_AXJO_new  = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, INDEX_AXJO_new, cols)
INDEX_FCHI_new  = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, INDEX_FCHI_new, cols)
INDEX_GDAXI_new = procces_stocks.align_date_in_dataframe(INDEX_DJIA_new, INDEX_GDAXI_new, cols)

##############################
#        DELETE COLS         #
##############################

cols = ['USD']
GOLD_new = procces_stocks.fill_gaps_with_interpolation(GOLD, GOLD_new, cols, ['linear'], False)
SILVER_new = procces_stocks.fill_gaps_with_interpolation(SILVER, SILVER_new, cols, ['linear'], False)
PLAT_new = procces_stocks.fill_gaps_with_interpolation(PLAT, PLAT_new, cols, ['linear'], False)
OIL_BRENT_new = procces_stocks.fill_gaps_with_interpolation(OIL_BRENT, OIL_BRENT_new, cols, ['linear'], False)

cols = ['Value']
USD_GBP_new = procces_stocks.fill_gaps_with_interpolation(USD_GBP, USD_GBP_new, cols, ['linear'], False)
JPY_USD_new = procces_stocks.fill_gaps_with_interpolation(JPY_USD, JPY_USD_new, cols, ['linear'], False)
AUD_USD_new = procces_stocks.fill_gaps_with_interpolation(AUD_USD, AUD_USD_new, cols, ['linear'], False)

cols = ['Open','High','Low','Close']
INDEX_HSI_new = procces_stocks.fill_gaps_with_interpolation(INDEX_HSI, INDEX_HSI_new, cols, ['linear'], False)
INDEX_IBEX_new = procces_stocks.fill_gaps_with_interpolation(INDEX_IBEX, INDEX_IBEX_new, cols, ['linear'], False)
INDEX_N225_new = procces_stocks.fill_gaps_with_interpolation(INDEX_N225, INDEX_N225_new, cols, ['linear'], False)
INDEX_AXJO_new  = procces_stocks.fill_gaps_with_interpolation(INDEX_AXJO, INDEX_AXJO_new, cols, ['linear'], False)
INDEX_FCHI_new  = procces_stocks.fill_gaps_with_interpolation(INDEX_FCHI, INDEX_FCHI_new, cols, ['linear'], False)
INDEX_GDAXI_new = procces_stocks.fill_gaps_with_interpolation(INDEX_GDAXI, INDEX_GDAXI_new, cols, ['linear'], False)

##############################
#        CHECK GAPS         #
##############################

cols = ['USD']
print procces_stocks.has_gaps(GOLD_new, cols, 0.0)
print procces_stocks.has_gaps(SILVER_new, cols, 0.0)
print procces_stocks.has_gaps(PLAT_new, cols, 0.0)
print procces_stocks.has_gaps(OIL_BRENT_new, cols, 0.0)

cols = ['Value']
print procces_stocks.has_gaps(USD_GBP_new, cols, 0.0)
print procces_stocks.has_gaps(JPY_USD_new, cols, 0.0)
print procces_stocks.has_gaps(AUD_USD_new, cols, 0.0)

cols = ['Open','High','Low','Close']
print procces_stocks.has_gaps(INDEX_DJIA_new, cols, 0.0)
print procces_stocks.has_gaps(INDEX_SP500_new, cols, 0.0)
print procces_stocks.has_gaps(INDEX_HSI_new, cols, 0.0)
print procces_stocks.has_gaps(INDEX_IBEX_new, cols, 0.0)
print procces_stocks.has_gaps(INDEX_N225_new, cols, 0.0)
print procces_stocks.has_gaps(INDEX_AXJO_new, cols, 0.0)
print procces_stocks.has_gaps(INDEX_FCHI_new, cols, 0.0)
print procces_stocks.has_gaps(INDEX_GDAXI_new, cols, 0.0)


##############################
#        CREATE COLS         #
##############################

shift_values = [1,2,3,5,10,20,30,50]
values_dfs = [GOLD_new, SILVER_new, PLAT_new, OIL_BRENT_new]
colnames = ['USD']
for df in values_dfs:
    df = procces_stocks.difference_between_consecutive_days(df, colnames, shift_values)
    df = procces_stocks.log_return(df, colnames, shift_values)

values_dfs = [USD_GBP_new, JPY_USD_new, AUD_USD_new]
colnames = ['Value']
for df in values_dfs:
    df = procces_stocks.difference_between_consecutive_days(df, colnames, shift_values)
    df = procces_stocks.log_return(df, colnames, shift_values)

values_dfs = [INDEX_DJIA_new, INDEX_HSI_new, INDEX_IBEX_new, INDEX_N225_new, INDEX_SP500_new, INDEX_AXJO_new, INDEX_FCHI_new, INDEX_GDAXI_new]
colnames = ['Open', 'Close', 'High', 'Low']
for df in values_dfs:
    df = procces_stocks.difference_between_consecutive_days(df, colnames, shift_values)
    df = procces_stocks.log_return(df, colnames, shift_values)

##############################
#          SAVE DATA         #
##############################

data_manipulation.write_csv_data(GOLD_new, '/Users/Pablo/Desktop/TFM/Data/GOLD.csv')
data_manipulation.write_csv_data(SILVER_new, '/Users/Pablo/Desktop/TFM/Data/SILVER.csv')
data_manipulation.write_csv_data(PLAT_new, '/Users/Pablo/Desktop/TFM/Data/PLAT.csv')
data_manipulation.write_csv_data(OIL_BRENT_new, '/Users/Pablo/Desktop/TFM/Data/OIL_BRENT.csv')

data_manipulation.write_csv_data(USD_GBP_new, '/Users/Pablo/Desktop/TFM/Data/USD_GBP.csv')
data_manipulation.write_csv_data(JPY_USD_new, '/Users/Pablo/Desktop/TFM/Data/JPY_USD.csv')
data_manipulation.write_csv_data(AUD_USD_new, '/Users/Pablo/Desktop/TFM/Data/AUD_USD.csv')

data_manipulation.write_csv_data(INDEX_DJIA_new, '/Users/Pablo/Desktop/TFM/Data/INDEX_DJIA.csv')
data_manipulation.write_csv_data(INDEX_HSI_new, '/Users/Pablo/Desktop/TFM/Data/INDEX_HSI.csv')
data_manipulation.write_csv_data(INDEX_IBEX_new, '/Users/Pablo/Desktop/TFM/Data/INDEX_IBEX.csv')
data_manipulation.write_csv_data(INDEX_N225_new, '/Users/Pablo/Desktop/TFM/Data/INDEX_N225.csv')
data_manipulation.write_csv_data(INDEX_SP500_new, '/Users/Pablo/Desktop/TFM/Data/INDEX_SP500.csv')
data_manipulation.write_csv_data(INDEX_AXJO_new, '/Users/Pablo/Desktop/TFM/Data/INDEX_AXJO.csv')
data_manipulation.write_csv_data(INDEX_FCHI_new, '/Users/Pablo/Desktop/TFM/Data/INDEX_FCHI.csv')
data_manipulation.write_csv_data(INDEX_GDAXI_new, '/Users/Pablo/Desktop/TFM/Data/INDEX_GDAXI.csv')