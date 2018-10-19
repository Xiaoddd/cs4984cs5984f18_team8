from sklearn import datasets
from sklearn.linear_model import LogisticRegression
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
    print(ratio_)
    # print(np.sum(ratio_))
    # print(len(mat))
    return pca.fit_transform(mat)

X = zscore_std(X)
X = pca_dim_red(X,20)
print(X)
clf = LogisticRegression(penalty='l1', solver='saga',
                                      tol=1e-6, max_iter=int(1e6),
                                      warm_start=True)
cvAcc = []
cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3, 16)
for c in cs:
    clf.set_params(C=c)
    # clf.fit(X, y)
    # coefs_.append(clf.coef_.ravel().copy())
    scores = cross_val_score(clf, X, y, cv=5)
    cvAcc.append(np.mean(scores))
    # print(cvAcc)
plt.plot(np.log10(cs), cvAcc, marker='o')
plt.show()


# # coefs_ = np.array(coefs_)
# # plt.plot(np.log10(cs), coefs_, marker='o')
# # ymin, ymax = plt.ylim()
# # plt.xlabel('log(C)')
# # plt.ylabel('Coefficients')
# # plt.title('Logistic Regression Path')
# # plt.axis('tight')
# # plt.show()
# # rng = np.random.RandomState(42)
# # X = np.c_[X, rng.randn(X.shape[0], 14)]  # add some bad features

# # # normalize data as done by Lars to allow for comparison
# # X /= np.sqrt(np.sum(X ** 2, axis=0))
# # keywords_selection(X,y)

# def keywords_selection(tr_input, tr_output, te_input=[], te_output=[]):
#   '''
#   input - matrix X \belongto R[n x p], where R denotes the real space, n is number of documents, p is the number of features (i.e., length of feature vector)
#   output - vector y \belongto R[n x 1], where R denotest the real space, n is number of documents
#   tr - denotes the training data set
#   te - denotes the testing data set
#   '''
#   # Five fold partition for cross-validation 
#   # Use regularized logistic regression for classification of the first 50 labeled documents
#   model = LassoCV(cv=5).fit(tr_input, tr_output)
#   # Display results
#   m_log_alphas = -np.log10(model.alphas_)
#   plt.figure()
#   ymin, ymax = 2300, 3800
#   plt.plot(m_log_alphas, model.mse_path_, ':')
#   plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
#            label='Average across the folds', linewidth=2)
#   plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
#               label='alpha: CV estimate')

#   plt.legend()

#   plt.xlabel('-log(alpha)')
#   plt.ylabel('Mean square error')
#   plt.axis('tight')
#   plt.ylim(ymin, ymax)