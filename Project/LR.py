import pandas as pd
import datetime
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import numpy as np
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LinearRegression





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


def MovingWindowMean(df):
    print (df)
    win = 16
    result = pd.DataFrame()
    df_size = len(df)
    for i in range(int(df_size/win)):
        if (i %500 == 0):
            print (i/(int(df_size/win))*100)
        if (i == (int(df_size/win) - 1)):
            temp = df.iloc[(i*win):]
        else:
            temp = df.iloc[(i*win):(i*win+win)]
        r = pd.DataFrame(temp.mean()).transpose()
        result = pd.concat([result,r])

    return result

def confusion_matrix(out, label):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(out)):
        if (out[i] == label[i]):
            if (out[i] == 1):
                tp += 1
            else:
                tn += 1
        else:
            if (out[i] == 1):
                fp += 1
            else:
                fn += 1
    return np.array([[tp,fp],[fn,tn]])



def precision(tp,fp):
    return float(tp) / (tp + fp)

def recall(tp,fn):
    return float(tp) / (tp + fn)

style.use('ggplot')    
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  

# stockFile = 'data_sets/data_batch1.csv'
# stock = pd.read_csv(stockFile)
# stock['UpdateTime'] = pd.to_datetime(stock['UpdateTime'].values)
# stock = stock.drop(['UpdateTime'], axis = 1)
# stock = MovingWindowMean(stock)
# stock.to_csv('data_batch1_after_resample_16s.csv')


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

# label = (midp_diff_arr < 0).astype(int)
# label = label*2 -1 

label = np.array(midp)
# print (label)


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
# print (data1)

# print (np.shape(data))
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c = 'r')
# # ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c = 'd')
# # plt.show()
# ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c = 'b')
# plt.show()

data,label = shuffle(data, label)
size = len(label)
train_data = data[0:int(0.8*size)]
train_label = label[0:int(0.8*size)]

test_data = data[int(0.8*size):int(1*size)]
test_label = label[int(0.8*size):int(1*size)]

ldf = pd.DataFrame(test_label)
# print (ldf[0].value_counts())

# gamma = 1.0 / np.size(train_data, 1)
# C = 1.0

# model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
# model.fit(train_data, train_label)
# print (model.score(test_data, test_label))
# out = model.predict(test_data)

# print (ldf[0].value_counts())
# print (confusion_matrix(out, test_label))

acc_col = []
pre_col = []
rec_col = []
id_col = []
f1_col = []
for i in range(1, 2):
    clf = LinearRegression()
    clf.fit(train_data, train_label)
    acc = clf.score(test_data, test_label)
    out = clf.predict(test_data)
    print (out)
    print (acc)
    # mat = confusion_matrix(out, test_label)
    # pre = precision(mat[0,0], mat[0,1])
    # rec = recall(mat[0,0], mat[1,0])
    # f1 = pre*rec*2/(pre+rec)
    # f1_col.append(f1)
    # acc_col.append(acc)
    # pre_col.append(pre)
    # rec_col.append(rec)
    # id_col.append(i)


# p1 = './fig/acc_randomforest'
# p2 = './fig/precision_randomforest'
# p3 = './fig/recall_randomforest'
# plt.plot(id_col, acc_col, 'b-')
# plt.title('acc_randomforest')
# plt.xlabel('n_estimatorss')
# plt.ylabel('acc_randomforest')
# plt.savefig(p1)
# plt.show()

# plt.plot(id_col, pre_col, 'b-')
# plt.title('precision_randomforest')
# plt.xlabel('n_estimatorss')
# plt.ylabel('precision_randomforest')
# plt.savefig(p2)
# plt.show()

# plt.plot(id_col, f1_col, 'b-')
# plt.title('f1_score_randomforest')
# plt.xlabel('n_estimatorss')
# plt.ylabel('f1_score_randomforest')
# plt.savefig(p2)
# plt.show()


# plt.plot(id_col, rec_col, 'r-')
# plt.title('recall_randomforest')
# plt.xlabel('n_estimatorss')
# plt.ylabel('recall_randomforest')
# plt.savefig(p3)
# plt.show()
