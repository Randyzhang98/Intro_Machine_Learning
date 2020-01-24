from SVM_SMO import SVM,KernelSVM,SoftMarginSVM
import csv
import numpy as np
import sklearn as sl
from sklearn import svm, datasets
from sklearn.utils import shuffle



if __name__ == '__main__':



    l2 = 'label_2.csv'
    s2 = 'sample_2.csv'
    with open(l2) as f:
        reader = csv.reader(f)
        ct_l2 = list(reader)
    ct_l2_np = np.array(ct_l2, dtype = float)
    with open(s2) as f:
        reader = csv.reader(f)
        ct_s2 = list(reader)
    ct_s2_np = np.array(ct_s2, dtype = float)

    ct_s2_np, ct_l2_np = shuffle(ct_s2_np, ct_l2_np, random_state = 10086)

    train_indices = np.random.choice(len(ct_s2_np),
                                 int(round(len(ct_s2_np)*0.8)),
                                 replace=False)
    test_indices = np.array(list(set(range(len(ct_s2_np))) - set(train_indices)))
    ct_s2_np_train = ct_s2_np[train_indices]
    ct_s2_np_test = ct_s2_np[test_indices]
    ct_l2_np_train = ct_l2_np[train_indices].astype(int)
    ct_l2_np_test = ct_l2_np[test_indices].astype(int)

    print (len(np.where(ct_l2_np_test == 1) [0] ))

    svm2 = KernelSVM(ct_s2_np_train, ct_l2_np_train)
    svm2.training(kernel = 'Gaussian', parameter=0.1)
    print (svm2.parameter_b())
    print (svm2.print_sv_num())
    acc2 = svm2.testing(ct_s2_np_test, ct_l2_np_test)
    print (acc2)

    M = svm.SVC(C = 100000000000, kernel='rbf', gamma=0.1)
    M.fit(ct_s2_np_train, ct_l2_np_train)
    print (M.n_support_)
    print (M.score(ct_s2_np_test, ct_l2_np_test))



