import numpy as np
from numpy import *
import csv

#载入数据
def loadData(filename):
    dataMat = []
    labelsMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            line = line.strip().split('\t')
            dataMat.append([line[0],line[1]])
            labelsMat.append(line[2])
 
    return dataMat,labelsMat
#选择J
def selectJrand(i,m):
    j = i
    while (i == j):
        j = int(np.random.uniform(0,m))
    return j
#限制范围
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj
#简化版的SMO
def smoSample(dataMatIn,classlabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn)
    labelsMat = mat(classlabels)
 
    b = 0
    m,n = shape(dataMatrix)
    alpha = mat(zeros((m,1))) #m个alpha 因为和数据的个数一样
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0 #记录alpha是否改变
        for i in range(m):#数据的个数
            fXi = float(multiply(alpha * labelsMat).T * dataMatrix*dataMatrix[i,:].T) + b  #书上124页的f（x）公式 #带入x求解
            Ei = fXi - float(labelsMat[i])  #误差，真实的标签与预测的误差
 
            if (Ei*labelsMat[i] < -toler and alpha[i] < C) or (Ei * labelsMat[i] > toler and alpha[i] > 0): #如果满足误差太大，alpha必须在范围内 需要优化alphai所以进行下面的
                j = selectJrand(i,m)
                fXj = float(multiply(alpha[j] * labelsMat[j]).T * dataMatrix*dataMatrix[j,:].T) + b #为对Xj的预测
                Ej = fXj - float(labelsMat[j])
                alphaIold = alpha[i].copy()
                alphaJold = alpha[j].copy()
                #限制范围 alphai 与 alpha j的
                if labelsMat[i] != labelsMat[j]:
                    L = max(0,alpha[j] - alpha[i])
                    H = min(C,C+ alpha[j] - alpha[i])
                else:
                    L = max(0,alpha[i] + alpha[j] -C)
                    H = min(C,alpha[i] + alpha[j])
 
                if L == H:
                    print('L == H')
                    continue #没法优化
 
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T \
                                                                - dataMatrix[j,:] * dataMatrix[j,:].T  #就是SMo的公式
                if eta > 0:
                    continue
 
                #对alphai 和alphaj 进行需改
                alpha[j] -= labelsMat[j]*(Ei - Ej)/eta
                alpha[j] = clip(alpha[j],H,L)
                if (abs(alpha[j] - alphaJold) < 0.0001):
                    print('alpha[j] 没有改的必要了')
                    continue
 
                alpha[i] += labelsMat[i] * labelsMat[j]/(alphaJold - alpha[j])
 
                #计算B 也是公式 自己博客里有
                b1 = b - Ei - labelsMat[i] * dataMatrix[i,:] * dataMatrix[i,:].T * (alpha[i] - alphaIold)\
                            - labelsMat[j] * dataMatrix[i,:] * dataMatrix[j,:].T * (alpha[j] - alphaJold)
                b2 = b - Ej - labelsMat[i] * dataMatrix[i,:] * dataMatrix[j,:].T * (alpha[i] - alphaIold)\
                            - labelsMat[j] * dataMatrix[j,:] * dataMatrix[j,:].T * (alpha[j] - alphaJold)
                if alpha[i] > 0 and alpha[i] < C:
                    b = b1
                if alpha[j] > 0 and alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
    return alpha,b

def calc_w (a, x, y):
    return np.dot(x.T, np.multiply(a,y))

if __name__ == '__main__':
    l1 = 'label_1.csv'
    s1 = 'sample_1.csv'
    with open(l1) as f:
        reader = csv.reader(f)
        ct_l1 = list(reader)
    ct_l1_np = np.array(ct_l1, dtype = float)
    with open(s1) as f:
        reader = csv.reader(f)
        ct_s1 = list(reader)
    ct_s1_np = np.array(ct_s1, dtype = float)

    x,y = ct_s1_np, ct_l1_np
