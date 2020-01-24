import pandas as pd
import datetime
import matplotlib.pylab as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# stockFile = 'data_sets/data_batch1.csv'
# stock = pd.read_csv(stockFile)
# stock['UpdateTime'] = pd.to_datetime(stock['UpdateTime'].values)
# stock = stock.drop(['UpdateTime'], axis = 1)
# stock = MovingWindowMean(stock)
# stock.to_csv('data_batch1_after_resample_16s.csv')


def LasDiff (midp, n = 1):    
    ind = 0
    L = []
    ind_l = []
    for i in midp:
        if ind == 0:
            old = i
        if ind < n:
            ind += 1
            continue
        ind_l.append(ind)
        l = i - old
        old = midp.iloc[ind-n]
        L.append(l)
        ind += 1
    return L



stock = pd.read_csv('data_batch1_after_resample.csv')
leng = len(stock)


# stock.index = stock['UpdateTime'].values
# print (stock.head(10))
# stock = stock.resample('1s').mean()

print (stock.head(10))

midp = stock.loc[:,'midPrice']
stock_train = midp
    
midp_diff = LasDiff(midp)

data = np.array(stock.iloc[:,0:108]).astype(float)
midp_diff_arr = np.array(midp_diff)

label = (midp_diff_arr < 0).astype(int)
label = label*2 -1 


ldf = pd.DataFrame(label)
print (ldf[0].value_counts())

data = data[:len(label)]

print (np.shape(data))
pca = PCA(n_components=3)
pca.fit(data)
data = pca.transform(data)

pf = True
nf = True
data1 =[]
data2 =  []
for i in range (len(data)):
    tmp = data[i]
    tl = label[i]
    if (tl == 1):
        data1.append(list(tmp))
    if (tl == -1):
        data2.append(list(tmp))

data1 = np.array(data1)
data2 = np.array(data2)
print (data1)

print (np.shape(data))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c = 'r')
# ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c = 'd')
# plt.show()
ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c = 'b')
plt.show()