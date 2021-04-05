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
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNNC


def Gini(p):
    return (p)*(1-(p)) + (1 -p)*(1 - (1-p))

def Entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2((1 - p))

def ClassError(p):
    return 1 - np.max([p, 1 - p])


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

# Decision Trees and impurity measures
            
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

x = np.arange(0.0, 1.0, 0.01)
ent = [Entropy(p) if p!= 0 else None for p in x]
scaled_ent = [e*0.5 if e else None for e in ent]
cl_err = [ClassError(i) for i in x]
gini = Gini(x)

fig1, ax1 = plt.subplots()

for i, lab, ls, c, in zip([ent, scaled_ent, gini, cl_err],
                          ['Entropy', 'Entropy (scaled)', 'Gini Impurity',
                           'Classification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green','cyan']):
    
    line = ax1.plot(x, i, label = lab, linestyle = ls, lw = 2, color = c)
    
ax1.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15),
           ncol = 5, fancybox = True, shadow = False)
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('impuriy index')
plt.title('Comparison of Impurities')
plt.tight_layout()

plt.show()
pause()

dtc = DTC(criterion = 'gini', max_depth = 4, random_state = 1)
dtc.fit(X_train, y_train)

plot_decision_regions(X, y, classifier = dtc, test_idx = range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc = 'upper left')
plt.title('Decision Tree with Gini Impurity')
plt.tight_layout()

plt.show()
pause()

tree.plot_tree(dtc)
plt.show()

pause()

rfc = RFC(criterion = 'gini', n_estimators = 25, random_state=1, n_jobs=2)
rfc.fit(X_train, y_train)
plot_decision_regions(X, y, classifier = rfc, test_idx = range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.title('Random Forest Classifier')
plt.legend(loc = 'upper left')
plt.tight_layout()

plt.show()
pause()

knn = KNNC(n_neighbors = 5, p = 2, metric = 'minkowski')
knn.fit(X_train, y_train)

plot_decision_regions(X, y, classifier = knn, test_idx = range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc = 'upper left')
plt.title('KNN')
plt.tight_layout()

plt.show()
pause()













