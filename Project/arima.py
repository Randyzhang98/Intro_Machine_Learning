import pandas as pd
import datetime
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

style.use('ggplot')    
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  

stockFile = 'data_sets/data_batch1.csv'
stock = pd.read_csv(stockFile)
print (stock.head(10))

midp = stock.loc[:,'midPrice']
stock_train = midp

# stock_train.plot(figsize=(12,8))
# plt.legend(bbox_to_anchor=(1.25, 0.5))
# plt.title("Stock Close")
# sns.despine()
# plt.show()

stock_diff = stock_train.diff()
stock_diff = stock_diff.dropna()

# plt.figure()
# plt.plot(stock_diff)
# plt.title('一阶差分')
# plt.show()



# abs_stock_diff = stock_diff.abs()
# me = abs_stock_diff.mean()
# ma = abs_stock_diff.max()
# hold = (me + (ma-me)*0.5)
# ind = []
# for i in range(len(abs_stock_diff)):
#     var = abs_stock_diff.iloc[i]
#     if (var >= hold):
#         t = [i]
#         t_var = stock_diff.iloc[i]
#         label = int (t_var > 0)*2 - 1
#         t.append(label)
#         ind.append(i)

# stock_diff = stock_diff.drop(ind)

# # initially want to eliminate the outlier, but it seems useless. 
# plt.figure()
# plt.plot(stock_diff)
# plt.title('一阶差分')
# plt.show()


# fig=plt.figure()
# ax1=fig.add_subplot(211)
# ax2=fig.add_subplot(212)
# acf = plot_acf(stock_diff, lags=20,ax=ax1,title="ACF")
# pacf = plot_pacf(stock_diff, lags=20,ax=ax2,title="PACF")
# plt.show()


# (p, q) =(sm.tsa.arma_order_select_ic(stock_train,max_ar=10,max_ma=10,ic='aic')['aic_min_order'])
#这里需要设定自动取阶的 p和q 的最大值，即函数里面的max_ar,和max_ma。ic 参数表示选用的选取标准，这里设置的为aic,当然也可以用bic。然后函数会算出每个 p和q 组合(这里是(0,0)~(3,3)的AIC的值，取其中最小的,这里的结果是(p=0,q=1)。

model = ARIMA(stock_train, order=(1, 1, 2))
result = model.fit()

pred = result.predict(dynamic=True, typ='levels')
print (pred)

plt.figure(figsize=(6, 6))
plt.xticks(rotation=45)
plt.plot(pred)
plt.plot(stock_train)
plt.show()