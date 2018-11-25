from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import l1_min_c
import numpy as np
# import matplotlib.pyplot as plt 
import json
# X = np.load("data/features_1023.npy")
X = np.load("data/features_1023.npy")
# X = np.delete(X,10,0)
y = np.genfromtxt('data/resp_100.csv', delimiter=',')
y = y[:,0]
print(np.shape(X))
test_keywords = ["access", "army", "bakkan", "campaign", "climate", "companies", "construction", "corp", "dakota",
                 "dapl", "donation", "drill", "duty", "energy", "fossil", "fuel", "indian", "indigenous", "keystone", "klp",
                 "mission", "missouri", "morton", "movement", "native", "nyemah", "obama", "oil", "nodapl", "patriots",
                 "pipeline","police", "protestor", "protest", "reservation", "resistance", "sacred", "sioux", "supplies", "tribal",
                 "tribe", "trump", "veteran", "violence", "volunteer", "water",'nodapl']
freq_keywords = ['pipeline', 'dakota', 'nodapl', 'social', 'media', 'oil', 'facebook', 'rock', 'people', 'movement', 
                'standing', 'youth', 'access', 'issue', 'gained', 'project', 'hashtags', 'protest', 'ground', 'twitter', 
                'bakken', 'iowa', 'protesters', 'large', 'instagram', 'energy', 'announced', 'attention', 'workers', 'patoka', 
                'young', 'area', 'transfer', 'eyes', 'north', 'construction', 'awareness', 'iron', 'youtube', 'sacred',
                 'indigenous', 'shortly', 'sioux', 'well', 'globe', 'september', '1,172-mile-long', '3.78', 'hashtag', 'teenage']
# y = np.delete(y,[10,49])
# Zscore
def zscore_std(mat, keys=test_keywords):
    mean_ = np.mean(mat[:101,:],axis=0)
    std_ = np.std(mat[:101,:], axis=0)
    pos_ = np.where(std_==0)
    mat = np.delete(mat, pos_, axis=1)
    mean_ = np.delete(mean_, pos_)
    std_ = np.delete(std_, pos_)
    meanMat = np.repeat([mean_],len(mat),axis=0)
    stdMat = np.repeat([std_],len(mat),axis=0)
    keys = np.delete(keys, pos_)
    return np.divide(mat-meanMat,stdMat), keys

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
def pen_logi_reg(covariates, response, penalty='l1',xlabels=test_keywords):
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
    # plt.xticks(np.arange(len(coefs_)),xlabels)
    plt.plot(np.arange(len(coefs_)),coefs_)
    plt.title('Model coefficients')
    plt.xlabel('Keywords')
    plt.ylabel('Coefficients')
    # plt.plot(np.log10(cs), cvAcc, marker='o')
    # plt.xlabel(test_keywords)
    print(np.where(np.abs(coefs_)>=1e-4))
    plt.show()
    return clf, cvAcc

if __name__ == "__main__":
    X_small = np.load("data/features_big.npy")
    X = np.concatenate((X, X_small), axis=0)
    # X = X[:,[ 6, 18, 21, 22, 26, 29, 31, 37]]
    X, xlabels = zscore_std(X)
    X_small = np.array(X[101:,:])
    X = np.array(X[:101,:])
    KFold = 5
    # X = pca_dim_red(X,20)
    # training
    train_idxs, test_idxs = kfold_partition(X, y, KFold)
    Training_Acc = list()
    Testing_Acc = list()
    Conf_Mat = list() 
    classificationResults = list()
    for i in range(KFold):
        print(f'{i+1}-th Fold cross validation:')
        X_train = X[np.array(train_idxs[i],dtype='int32'),:]
        y_train = y[np.array(train_idxs[i],dtype='int32')]
        X_test = X[np.array(test_idxs[i],dtype='int32'),:]
        y_test = y[np.array(test_idxs[i],dtype='int32')]
        
        mdl,acc = pen_logi_reg(X_train, y_train, penalty='l1', xlabels=xlabels)
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
        yhat_small = mdl.predict(X_small)
        classificationResults.append(yhat_small)
        # print(mdl.intercept_)
        # print(mdl.coef_)
        # print(np.where(mdl.coef_[0]!=0))
    print(np.mean(np.array(Training_Acc)))
    print(np.mean(np.array(Testing_Acc)))
np.save('preprocessed/Classification_Results_100.npy', Training_Acc, Testing_Acc, Conf_Mat)
np.save('preprocessed/Classifying_Big',classificationResults,Training_Acc, Testing_Acc)

## Organizing classified labels for small corpus and ignore the irrelevant documents
classifiedLabel = np.load('preprocessed/Classifying_Big.npy')
labels = classifiedLabel[0]
np.savetxt("preprocessed/labels_big.csv",labels,delimiter=",")
pos = np.array(np.where(labels==1),dtype="int").tolist()
np.savetxt('preprocessed/rel_id_start_from_0.csv',pos[0],delimiter=",")
NoDAPL_file = "data/part-00000-66d9f78f-37f9-4dea-985c-6e2c040632ef-c000.json"
with open(NoDAPL_file, 'r') as data_file:
    data = json.load(data_file)
    relCorpus = [data[i] for i in pos[0]]
print(len(relCorpus))
with open('preprocessed/big_relevant.json', 'w') as outfile:
    json.dump(relCorpus,outfile)
