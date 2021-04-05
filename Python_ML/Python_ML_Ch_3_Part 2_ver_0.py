# -*- coding: utf-8 -*-
"""
Chapter 3 Examples from Python Machine Learning
Part 2
@author: Wes User
"""

import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score as acc_sc
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def pause():
    sys.stdout.flush()
    input("Hit Enter to continue")

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
            
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                    random_state = 1, stratify = y)

print('Data Set:')
print('X size = ', len(X))
print('X test size =', len(X_test))
print('y size = ', np.bincount(y))
print('y test size = ', np.bincount(y_test))
print()

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))

np.random.seed(1)

X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0) # True or False
                       
y_xor = np.where(y_xor, 1, -1) # 1 or -1

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c ='b', marker = 'x',
            label = '1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c ='r', marker = 's',
            label = '-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.tight_layout()

plt.show()
pause()

svm = SVC(kernel = 'rbf', random_state = 1, gamma = .1, C= 100.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier = svm)
plt.legend(loc = 'upper left')
plt.tight_layout()

plt.show()
pause()

svm2 = SVC(kernel = 'rbf', random_state = 1, gamma = 10.0, C= 1.0)
svm2.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y, classifier = svm2,
                      test_idx = range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title('SVC - Gamma = 10, C = 1.0')
plt.legend(loc = 'upper left')
plt.tight_layout()

plt.show()
pause()


