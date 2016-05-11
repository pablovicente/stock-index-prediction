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
print(len(x))
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