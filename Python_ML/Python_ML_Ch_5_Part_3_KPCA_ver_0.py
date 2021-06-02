# -*- coding: utf-8 -*-
"""

Python Machine Learning - Ch. 5
Kernel Principal Component Analysis

Deviated some by skipping concentric circles,
added SVM classification and split into
Train and Test sets with total n = 1500.

"""

import sys
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from sklearn.datasets import make_moons  #, make_circles
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split as T_T_S
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

def pause():
    sys.stdout.flush()
    input("Hit Enter to continue")

def rbf_kernel_pca(X, gamma, n_components):
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = np.exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    X_pc = np.column_stack([eigvecs[:, i] for i in range(n_components)])

    return X_pc

def rbf_kernel_pcaEV(X, gamma, n_components):
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = np.exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    alphas = np.column_stack([eigvecs[:, i] for i in range(n_components)])
    lambdas = [eigvals[i] for i in range(n_components)]
    

    return alphas, lambdas

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):

    markers = ['s', 'x', 'o', '^', 'v']
    colors = ['r', 'b', 'g', 'm', 'c']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:,0].min() - 1, X[:, 0].max() + 1   
    x2_min, x2_max = X[:,1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))

    Z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl,1],
                    alpha = 0.7, c=colors[idx],
                    marker = markers[idx], label = cl,
                    edgecolor='black')
        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], c='y',edgecolor = 'black',
                        alpha = 0.4, linewidth = 1, marker ='o',
                        s=100, label = 'test set')


print('Plot of the data')
print()
X, y = make_moons(n_samples = 1500, random_state = 123)
X_train, X_test, y_train, y_test =\
    T_T_S(X,y,test_size = .3333, random_state = 0, stratify = y)


plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
            color = 'red', marker ='^', alpha = 0.5)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
            color = 'blue', marker ='o', alpha = 0.5)
plt.title('Half moon - training data n = 1000')
plt.tight_layout()
plt.show()

pause()

print('Use Scikit-Learn PCA to try to separate. ')
print()
scikit_pca = PCA(n_components =  2)
X_spca_tr = scikit_pca.fit_transform(X_train)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (7,3))
ax[0].scatter(X_spca_tr[y_train==0, 0], X_spca_tr[y_train==0, 1],
              color='red', marker='^', alpha = 0.5)
ax[0].scatter(X_spca_tr[y_train==1, 0], X_spca_tr[y_train==1, 1],
              color='blue', marker='o', alpha = 0.5)
ax[1].scatter(X_spca_tr[y_train==0, 0], np.zeros((500,1)) + .02,
              color='red', marker='^', alpha = 0.5)
ax[1].scatter(X_spca_tr[y_train==1, 0], np.zeros((500,1)) - 0.02,
              color='blue', marker='o', alpha = 0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[0].set_title("Scikit-Learn PCA, n components = 2")
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[1].set_title("n components = 1")
plt.tight_layout()
plt.show()

pause()

print('Kernel principal component analysis ')
print()

X_kpca = rbf_kernel_pca(X_train, gamma = 10.,
                                                      n_components = 2)
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].scatter(X_kpca[y_train==0, 0], X_kpca[y_train==0, 1],
              color='red', marker='^', alpha = 0.5)
ax[0].scatter(X_kpca[y_train==1, 0], X_kpca[y_train==1, 1],
              color='blue', marker='o', alpha = 0.5)
ax[1].scatter(X_kpca[y_train==0, 0], np.zeros((500,1)),
              color='red', marker='^', alpha = 0.5)
ax[1].scatter(X_kpca[y_train==1, 0], np.zeros((500,1)),
              color='blue', marker='o', alpha = 0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[0].set_title('Kernel PCA n components = 2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[1].set_title('n components = 1')
plt.tight_layout()
plt.show()

pause()

print('Use Support Vector Machine and plot decision boundary ')
print()

kpca_svm = SVC(kernel = 'poly', degree = 2, coef0 = 1, random_state=1, C = 100.0)
kpca_svm.fit(X_kpca, y_train) # now using class for classification
plot_decision_regions(X_kpca, y_train, classifier = kpca_svm, resolution = .005)

plt.xlim(-.1, .1)
plt.ylim(-.1, .1)
plt.title('Support Vector Machine - kernel = poly, degree = 2')
plt.tight_layout()
plt.show()

pause()

print('Use the eigenvalues/vectors from training set on test set and plot ')
print()

rows_test = len(X_test)
rows_train = len(X_train)
X_kpca_test = np.zeros((rows_test,2))

alphas, lambdas = rbf_kernel_pcaEV(X_train, gamma = 15, n_components = 2)
gamma = 15

for i in range(rows_test):
    pair_dist = np.array([np.sum((X_test[i] - row)**2) for row in X_train])
    k = np.exp(-gamma * pair_dist)
    X_kpca_test[i] = -( k.dot(alphas/lambdas))  #added miinus sign
    
    
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(X_kpca_test[y_test==0, 0], X_kpca_test[y_test==0, 1],
              color='red', marker='^', alpha = 0.5)
ax.scatter(X_kpca_test[y_test==1, 0], X_kpca_test[y_test==1, 1],
              color='blue', marker='o', alpha = 0.5)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_ylim([-.1,.1])
ax.set_xlim([-.1,.1])
ax.set_title('KPCA test data - n = 500 ')
plt.tight_layout()
plt.show()

pause() 
   
score = kpca_svm.score(X_kpca_test, y_test)
print('Use the SVM to predict the test data after kpca ')
print('Score = ', score)
print()

y_pred = kpca_svm.predict(X_kpca_test)

pause()

print('Now use the Scikit-Learn KPCA tools ')
print()

scikit_kpca = KernelPCA(n_components = 2, kernel = 'rbf', gamma = 15)
X_SK = scikit_kpca.fit_transform(X_train)
X_SK = X_SK * [1,-1] # flip on y axis

plt.scatter(X_SK[y_train==0, 0], X_SK[y_train==0, 1], 
            color = 'red', marker ='^', alpha = 0.5)
plt.scatter(X_SK[y_train==1, 0], X_SK[y_train==1, 1], 
            color = 'blue', marker ='o', alpha = 0.5)
plt.xlim(-1.,1.)
plt.ylim(-1.,1.)
plt.title('Scikit-Learn KPCA - training data n=1000, n components = 2 ')
plt.tight_layout()
plt.show()

print('--- Done ---')
print()



