# This is the template of VE445 JI 2019 Spring Lab 1
# Name: Zhang, Sijun
#   ID: 516370910155
# Date: March 19th, 2019

import numpy as np
import random as rnd
import os
from numpy import *

MAX_FLOAT = 10000000.0
MAX_ITER = 20000



class SVM(object):
    # This class is the hard margin SVM and it is the parent
    # class of KernelSVM and SoftMarginSVM.
    # Please add any function to the class if it is needed.
    def __init__(self, sample, label):
    # This function is an constructor and shouldn't be modified.
    # The 'self.w' represents the director vector and should be
    # in the form of numpy.array 
    # The 'self.b' is the displacement of the SVM and it should
    # be a float64 variable.
        self.dim = sample.shape[1]
        self.sample = sample
        self.label =label
        self.num = self.label.shape[0]
        self.a = np.zeros((self.num))
        self.w = np.zeros((self.dim))
        self.b = 0.0
        self.C = float ('inf')
        self.min_loss = 0.001
        self.max_iter = MAX_ITER  
        self.kernel = 'Linear'
        self.parameter = 0.0

    def rand_index(self,l, h, i):
        j = i
        while j == i:
            j = rnd.randint(l,h-1)
        return j

    def training(self):
    # Implement this function by yourself and do not modifiy 
    # the parameter.
        x = self.sample
        y = self.label
        y = (y.T[0])
        # self.kernel = kernel
        # self.parameter = parameter
        K = self.Linear(x, x)
        # print (K)
        ite = 0
        passes = 0
        max_passes = 3
        while ite < self.max_iter:
            num_changed_a = 0
            ite += 1
            print ('ite  = '+ str(ite))
            for i in range (self.num):
                t = (self.a * y)
                Ei = np.dot(t, K[i] )+ self.b - y[i]
                if (y[i]*Ei < -self.min_loss and self.a[i] < self.C) or (y[i]*Ei > self.min_loss and self.a[i] > 0):
                    j = self.rand_index(0, self.num, i)
                    Ej = np.dot(t, K[j] )+ self.b - y[j]
                    aio = self.a[i]
                    ajo = self.a[j]
                    if y[i] != y[j]:
                        L = max(0, self.a[j] - self.a[i])
                        H = MAX_FLOAT
                    else:
                        L = 0
                        H = self.a[i] + self.a[j]
                    if L==H:
                        continue
                    eta = 2*K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0: 
                        continue
                    ajn = self.a[j] - y[j]*(Ei-Ej)/eta
                    if ajn > H:
                        self.a[j] = H
                    elif ajn < L:
                        self.a[j] = L
                    else:
                        self.a[j] = ajn
                    if (self.a[j] - ajo < 0.00001 and self.a[j] - ajo > -0.00001):
                        continue
                    self.a[i] = self.a[i] + y[i]*y[j]*(ajo-self.a[j])
                    
                    #b part
                    b1 = self.b - Ei - y[i]*(self.a[i] - aio)*K[i,i] - y[j]*(self.a[j] - ajo) * K[i,j]
                    b2 = self.b - Ei - y[i]*(self.a[i] - aio)*K[i,j] - y[j]*(self.a[j] - ajo) * K[j,j]

                    if (self.a[i] > 0 and self.a[i] < self.C):
                        self.b = b1
                    elif (self.a[j] > 0 and self.a[j] < self.C):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) /2.0
                    num_changed_a = num_changed_a + 1
            if num_changed_a <= 3:
                passes += 1
            else:
                passes = 0
            print ('num_changed_a = ' + str(num_changed_a))
            if (passes >= max_passes):
                break
        self.w = np.dot(x.T, np.multiply(self.a, y))
    

    def Linear(self, x, y):
        return np.matmul(x, y.T)    
        
    def testing(self, test_sample, test_label):
    # This function should return the accuracy 
    # of the input test_sample in float64, e.g 0.932
        y_vat = np.sign(np.dot(self.w.T, test_sample.T) +self.b).astype(int)
        y = (np.transpose(test_label)[0])
        # print (y)
        idx = np.where(y_vat == 1)
        tp = np.sum( abs( y[idx] - y_vat[idx] < 0.0001) )
        idx = np.where(y_vat == -1)

        tn = np.sum( abs( y[idx] - y_vat[idx] < 0.0001))
        
        return float(tp + tn ) / len( y)

    def print_sv_num(self):
        cnt = [0,0]
        for i in range(len(self.a)):
            if (self.a[i] != 0) & (self.label[i] == 1):
                cnt[0] += 1
            elif (self.a[i] != 0) & (self.label[i] == -1):
                cnt[1] +=1
        return cnt


    def parameter_w(self):
    # This function is used to return the parameter w of the SVM.
    # The result is supposed to be an np.array
    # This functin shouldn't be modified.
        return self.w
    def parameter_b(self):
    # This function is used to return the parameter self.b of the SVM.
    # The result is supposed to be an real number.
    # This functin shouldn't be modified.
        return self.b  

    # If you choose to use tf.InteractiveSession, please remember 
    # to close it or there might be memory overflow.
    # You can recycle the resource by using a destructor.


class KernelSVM(SVM):
    # This class is the kernel SVM.
    # Please add any function to the class if it is needed.

    def training(self, kernel = 'Linear', parameter = 1):
    # Specifics:
    #   For the parameter of 'kernel':
    #   1. The default kernel function is 'Linear'.
    #      The parameter is 1 by default.
    #   2. Gaussian kernel function is 'Gaussian'.
    #      The parameter is the Gaussian bandwidth.
    #   3. Laplace kernel funciton is 'Laplace'.
    #   4. Polynomial kernel functino is 'Polynomial'.
    #      The parameter is the exponential of polynomial.
    # Add your cold after the initialization.
        self.kernel = kernel
        self.parameter = parameter
        x = self.sample
        y = self.label
        y = (y.T[0])
        # self.kernel = kernel
        # self.parameter = parameter
        if self.kernel == 'Linear':
            K = self.Linear(x,x)
        elif self.kernel == 'Polynomial':
            K = self.Polynomial(x,x, self.parameter)
        elif self.kernel == 'Gaussian':
            K = self.Gaussian(x,x,self.parameter)
        elif self.kernel == 'Laplace':
            K = self.Laplace(x,x, self.parameter)

        # print (K)
        ite = 0
        passes = 0
        max_passes = 3
        while ite < self.max_iter:
            num_changed_a = 0
            ite += 1
            print ('ite  = '+ str(ite))
            for i in range (self.num):
                t = (self.a * y)
                # print (t)
                # print (y.shape)
                Ei = np.dot(t, K[i] )+ self.b - y[i]
                # print (y[i])
                # print (Ei)
                # print (self.a[i])
                if (y[i]*Ei < -self.min_loss and self.a[i] < self.C) or (y[i]*Ei > self.min_loss and self.a[i] > 0):
                    j = self.rand_index(0, self.num, i)
                    Ej = np.dot(t, K[j] )+ self.b - y[j]
                    aio = self.a[i]
                    ajo = self.a[j]
                    if y[i] != y[j]:
                        L = max(0, self.a[j] - self.a[i])
                        H = MAX_FLOAT
                    else:
                        L = 0
                        H = self.a[i] + self.a[j]
                    if L==H:
                        continue
                    eta = 2*K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0: 
                        continue
                    ajn = self.a[j] - y[j]*(Ei-Ej)/eta
                    if ajn > H:
                        self.a[j] = H
                    elif ajn < L:
                        self.a[j] = L
                    else:
                        self.a[j] = ajn
                    if (self.a[j] - ajo < 0.00001 and self.a[j] - ajo > -0.00001):
                        continue
                    self.a[i] = self.a[i] + y[i]*y[j]*(ajo-self.a[j])
                    
                    #b part
                    b1 = self.b - Ei - y[i]*(self.a[i] - aio)*K[i,i] - y[j]*(self.a[j] - ajo) * K[i,j]
                    b2 = self.b - Ei - y[i]*(self.a[i] - aio)*K[i,j] - y[j]*(self.a[j] - ajo) * K[j,j]

                    if (self.a[i] > 0 and self.a[i] < self.C):
                        self.b = b1
                    elif (self.a[j] > 0 and self.a[j] < self.C):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) /2.0
                    num_changed_a = num_changed_a + 1
            if num_changed_a <= 3:
                passes += 1
            else:
                passes = 0
            print ('num_changed_a = ' + str(num_changed_a))
            if (passes >= max_passes):
                break
        if (self.kernel == 'Linear'):
            self.w = np.dot(x.T, np.multiply(self.a, y))


    def testing(self, test_sample, test_label):
        y = test_label
        if (len(y) >=1000 ):
            y = (y.T[0])   
        y_train = np.transpose(self.label)[0]     
        if self.kernel == 'Linear':
            y_vat = np.sign(np.dot(self.w.T, test_sample.T) +self.b).astype(int)
        elif self.kernel == 'Polynomial':
            y_vat = np.sign(np.add(np.matmul(np.multiply(self.a, y_train), self.Polynomial(self.sample, (test_sample), self.parameter)), self.b)).astype(int)
        elif self.kernel == 'Gaussian':
            rA = np.reshape(np.square(self.sample).sum(axis = 1),[-1,1])
            rB = np.reshape(np.square(test_sample).sum(axis = 1),[-1,1])
            print ('in gaussian predict')
            sq_dists = np.sqrt(np.abs(np.add(np.subtract(rA, np.multiply(0.02, np.matmul(self.sample, np.transpose(test_sample)))), np.transpose(rB))))
            print (sq_dists)
            temp = np.exp(-sq_dists/(2*pow(self.parameter, 2)))
            print (temp)
            o = np.add(np.matmul(np.multiply(self.a, y_train), temp), self.b)
            y_vat = np.sign(o).astype(int)
        elif self.kernel == 'Laplace':
            gamma = self.parameter
            rA = np.reshape(np.square(self.sample).sum(axis = 1),[-1,1])
            rB = np.reshape(np.square(test_sample).sum(axis = 1),[-1,1])
            sq_dists = np.sqrt(np.abs(np.add(np.subtract(rA, np.multiply(2, np.matmul(self.sample, np.transpose(test_sample)))), np.transpose(rB))))
            temp = np.exp(-sq_dists/gamma)
            y_vat = np.sign(np.add(np.matmul(np.multiply(self.a, y_train), temp), self.b)).astype(int)
        
        print (len (np.where(y_vat == 1)[0]) )
        print (y)
        print (y_vat)

        idx = np.where(y_vat == 1)
        tp = np.sum( abs( y[idx] - y_vat[idx] < 0.0001) )
        idx = np.where(y_vat == -1)

        tn = np.sum( abs( y[idx] - y_vat[idx] < 0.0001))
        
        return float(tp + tn ) / len( y)


    def Polynomial(self, x, y, d):
        return pow(np.matmul(x,y.T),d )
    def Gaussian(self, x, y, gamma):
        dist = np.reshape(np.square(x).sum(axis = 1), [-1,1] )
        sq_d = np.sqrt(np.abs(np.add(np.subtract(dist, np.multiply(2, np.matmul(x, y.T))), np.transpose(dist))))
        return (-sq_d/(2*(gamma**2)))
    def Laplace(self, x, y, gamma):
        dist = np.reshape(np.square(x).sum(axis = 1), [-1,1] )
        sq_d = np.sqrt(np.abs(np.add(np.subtract(dist, np.multiply(2, np.matmul(x, y.T))), np.transpose(dist))))
        return (-sq_d/gamma)

class SoftMarginSVM(KernelSVM):
    # This class is the soft margin SVM and inherits
    # the kernel SVM to expand to both linear Non-seperable and
    # soft margin problem.
    # Please add any function to the class if it is needed.
    def training(self, kernel = 'Linear', parameter = 1, C = 1.0):
    # Specifics:
    #   For the parameter of 'kernel':
    #   1. The default kernel function is 'Linear'.
    #      The parameter is 1 by default.
    #   2. Gaussian kernel function is 'Gaussian'.
    #      The parameter is the Gaussian bandwidth.
    #   3. Laplace kernel funciton is 'Laplace'.
    #   4. Polynomial kernel functino is 'Polynomial'.
    #      The parameter is the exponential of polynomial.
    # Add your cold after the initialization.
        self.C = C
        self.kernel = kernel
        self.parameter = parameter
        x = self.sample
        y = self.label
        if (len(y) >=1000 ):
            y = (y.T[0])
        # self.kernel = kernel
        # self.parameter = parameter
        if self.kernel == 'Linear':
            K = self.Linear(x,x)
        elif self.kernel == 'Polynomial':
            K = self.Polynomial(x,x, self.parameter)
        elif self.kernel == 'Gaussian':
            K = self.Gaussian(x,x,self.parameter)
        elif self.kernel == 'Laplace':
            K = self.Laplace(x,x, self.parameter)

        # print (K)
        ite = 0
        passes = 0
        max_passes = 3
        while ite < self.max_iter:
            num_changed_a = 0
            ite += 1
            print ('ite  = '+ str(ite))
            for i in range (self.num):
                t = (self.a * y)
                # print (t)
                # print (y.shape)
                Ei = np.dot(t, K[i] )+ self.b - y[i]
                # print (y[i])
                # print (Ei)
                # print (self.a[i])
                if (y[i]*Ei < -self.min_loss and self.a[i] < self.C) or (y[i]*Ei > self.min_loss and self.a[i] > 0):
                    j = self.rand_index(0, self.num, i)
                    Ej = np.dot(t, K[j] )+ self.b - y[j]
                    aio = self.a[i]
                    ajo = self.a[j]
                    if y[i] != y[j]:
                        L = max(0, self.a[j] - self.a[i])
                        H = min(C, C + self.a[j] - self.a[i])
                    else:
                        L = max(0, self.a[i] + self.a[j] - C)
                        H = min(C, self.a[i] + self.a[j])
                    if L==H:
                        continue
                    eta = 2*K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0: 
                        continue
                    ajn = self.a[j] - y[j]*(Ei-Ej)/eta
                    if ajn > H:
                        self.a[j] = H
                    elif ajn < L:
                        self.a[j] = L
                    else:
                        self.a[j] = ajn
                    if (self.a[j] - ajo < 0.00001 and self.a[j] - ajo > -0.00001):
                        continue
                    self.a[i] = self.a[i] + y[i]*y[j]*(ajo-self.a[j])
                    
                    #b part
                    b1 = self.b - Ei - y[i]*(self.a[i] - aio)*K[i,i] - y[j]*(self.a[j] - ajo) * K[i,j]
                    b2 = self.b - Ei - y[i]*(self.a[i] - aio)*K[i,j] - y[j]*(self.a[j] - ajo) * K[j,j]

                    if (self.a[i] > 0 and self.a[i] < self.C):
                        self.b = b1
                    elif (self.a[j] > 0 and self.a[j] < self.C):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) /2.0
                    num_changed_a = num_changed_a + 1
            if num_changed_a <= 3:
                passes += 1
            else:
                passes = 0
            print ('num_changed_a = ' + str(num_changed_a))
            if (passes >= max_passes):
                break
        if (self.kernel == 'Linear'):
            self.w = np.dot(x.T, np.multiply(self.a, y))
    