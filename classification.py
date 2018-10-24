from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import l1_min_c
import numpy as np
import matplotlib.pyplot as plt 

X = np.load("data/features_1023.npy")
# X = np.delete(X,10,0)
y = np.genfromtxt('data/resp_100.csv', delimiter=',')
y = y[:,0]
print(np.shape(X))

# y = np.delete(y,[10,49])
# Zscore
def zscore_std(mat):
    mean_ = np.mean(mat,axis=0)
    std_ = np.std(mat, axis=0)
    pos_ = np.where(std_==0)
    mat = np.delete(mat, pos_, axis=1)
    mean_ = np.delete(mean_, pos_)
    std_ = np.delete(std_, pos_)
    meanMat = np.repeat([mean_],len(y),axis=0)
    stdMat = np.repeat([std_],len(y),axis=0)
    return np.divide(mat-meanMat,stdMat)

# PCA dimension reduction
def pca_dim_red(mat, n_comp=20):
    pca = PCA(n_components=n_comp)
    pca.fit(mat)
    ratio_ = pca.explained_variance_ratio_
    # print(ratio_)
    # print(np.sum(ratio_))
    # print(len(mat))
    return pca.fit_transform(mat)

# Cross validation fold partition
def kfold_partition(input, resp, K):
    trainIdxs = list()
    testIdxs = list()
    n = len(resp)
    randPerm = np.random.permutation(n)
    n_fold = int(np.floor(n/K))
    print(n_fold)
    folds = list()
    for i in range(K):
        if i<(K-1):
            folds.append(np.array(randPerm[i*n_fold:(i+1)*n_fold]))
        else: 
            folds.append(np.array(randPerm[i*n_fold:]))
    for i in range(K):
        tmp = np.array([])
        for j in range(K):
            if j!=i:
                tmp = np.concatenate((tmp,folds[j]))
        trainIdxs.append(tmp)
        testIdxs.append(folds[i])
    return trainIdxs, testIdxs

# Penalized Logistic regression with cross validation
def pen_logi_reg(covariates, response, penalty='l1'):
    clf = LogisticRegression(penalty='l1', solver='saga', tol=1e-6, max_iter=int(1e6), warm_start=True,fit_intercept=True)
    cvAcc = list()
    coefs_ = list()
    cs = l1_min_c(covariates, response, loss='log') * np.logspace(0, 2, 16)
    for c in cs:
        clf.set_params(C=c)
        # clf.fit(X, y)
        # coefs_.append(clf.coef_.ravel().copy())
        scores = cross_val_score(clf, covariates, response, cv=5)
        cvAcc.append(np.mean(scores))
        # print(cvAcc)
    cvAcc = np.array(cvAcc)
    min = np.amax(cvAcc)
    pos = np.where(cvAcc==min)
 
    clf.set_params(C=cs.item(pos[0].item(0)))
    clf.fit(covariates,response)
    coefs_ = clf.coef_.ravel().copy()
    print('Model coefficients: ')
    print(coefs_)
    # plt.plot(np.arange(len(coefs_)),coefs_)
    # plt.title('Best coefficients')
    # # plt.plot(np.log10(cs), cvAcc, marker='o')
    # plt.show()
    return clf, cvAcc

if __name__ == "__main__":
    # X = X[:,[ 6, 18, 21, 22, 26, 29, 31, 37]]
    X = zscore_std(X)
    KFold = 5
    # X = pca_dim_red(X,20)
    # training
    train_idxs, test_idxs = kfold_partition(X, y, KFold)
    Training_Acc = list()
    Testing_Acc = list()
    Conf_Mat = list() 
    for i in range(KFold):
        print(f'{i+1}-th Fold cross validation:')
        X_train = X[np.array(train_idxs[i],dtype='int32'),:]
        y_train = y[np.array(train_idxs[i],dtype='int32')]
        X_test = X[np.array(test_idxs[i],dtype='int32'),:]
        y_test = y[np.array(test_idxs[i],dtype='int32')]
        
        mdl,acc = pen_logi_reg(X_train, y_train, penalty='l1')
        print('Training cross validation: ')
        print(acc)
        print('Traing accuracy: ')
        print(mdl.score(X_train, y_train))
        print('Testing accuracy: ')
        print(mdl.score(X_test, y_test))
        yhat = mdl.predict(X_test)
        print(confusion_matrix(y_test,yhat))
        tn, fp, fn, tp = confusion_matrix(y_test,yhat).ravel()
        print([tn, fp, fn, tp])
        Training_Acc.append(mdl.score(X_train, y_train))
        Testing_Acc.append(mdl.score(X_test, y_test))
        Conf_Mat.append(confusion_matrix(y_test,yhat))
        # print(mdl.intercept_)
        # print(mdl.coef_)
        # print(np.where(mdl.coef_[0]!=0))
    print(np.mean(np.array(Training_Acc)))
    print(np.mean(np.array(Testing_Acc)))
np.save('preprocessed/Classification_Results_100.npy', Training_Acc, Testing_Acc, Conf_Mat)