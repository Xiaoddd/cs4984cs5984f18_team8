from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import l1_min_c
import numpy as np
import matplotlib.pyplot as plt 
X = np.load("data/features1016.npy")
y = np.genfromtxt('data/resp.csv', delimiter=',')
print(y)
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

# Penalized Logistic regression with cross validation
def pen_logi_reg(covariates, response, penalty='l1'):
    clf = LogisticRegression(penalty='l1', solver='saga', tol=1e-6, max_iter=int(1e6), warm_start=True,fit_intercept=True)
    cvAcc = list()
    coefs_ = list()
    cs = l1_min_c(covariates, response, loss='log') * np.logspace(0, 1, 16)
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
    # print(coefs_)
    # plt.plot(np.arange(len(coefs_)),coefs_)
    # plt.title('Best coefficients')
    # # plt.plot(np.log10(cs), cvAcc, marker='o')
    # plt.show()
    return clf, cvAcc

if __name__ == "__main__":
    X = X[:,[ 6, 18, 21, 22, 26, 29, 31, 37]]
    X = zscore_std(X)
    # X = pca_dim_red(X,20)
    mdl,acc = pen_logi_reg(X, y, penalty='l2')
    print(mdl.score(X,y))
    print(acc)
    yhat = mdl.predict(X)
    print(confusion_matrix(y,yhat))
    tn, fp, fn, tp = confusion_matrix(y,yhat).ravel()
    print([tn, fp, fn, tp])
    # print(mdl.intercept_)
    # print(mdl.coef_)
    # print(np.where(mdl.coef_[0]!=0))
