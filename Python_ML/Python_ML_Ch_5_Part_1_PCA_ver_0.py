# -*- coding: utf-8 -*-
"""

Python Machine Learning Chapter 5
Principal Component Analysis
Scikit-Learn PCA added at end.

"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as T_T_S
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score as ACC_SC
from sklearn.linear_model import LogisticRegression as LR
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

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

df_wine = pd.read_csv('wine.txt')
df_wine.columns = ['Class label', 'Alcohol',
                'Malic Acid', 'Ash',
                'Alcalinity of ash', 'Magnesium',
                'Total phenols', 'Flavanoids',
                'Nonflavanoids phenols', 'Proanthocyanins',
                'Color Intensity', 'Hue',
                'OD280/OD315 of diluted wines',
                'Proline']

print('Wine data')
print()
print(df_wine.head())
print()
print(df_wine.tail())
print()
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =\
    T_T_S(X,y,test_size = .3, random_state = 0, stratify = y)
# stratify ensures same class proportions of training and test data sets

print('Training Data Size = ' , len(X_train))
print('Test Data Size = ' , len(X_test))
print()

pause()

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
# Through step 1
cov_mat = np.cov(X_train_std.T)
# through step 2
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# through step 3
tot = sum(eigen_vals)
var_exp = [(i /tot) for i in sorted(eigen_vals, reverse = True)]
cum_var = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, alpha = 0.5, align = 'center',
        label = 'Individual explained variance')
plt.step(range(1,14), cum_var, where = 'mid',
         label = 'Cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

pause()

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
eigen_pairs.sort(key = lambda k: k[0], reverse = True)
W = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))

# W is the eigenvectors associated with the two largest (abs) eigenvalues
# 2 is selected so as to plot out.
#
X_train_pca = X_train_std.dot(W)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label = l, marker = m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA with k = 2, note this is unsupervised learning - labels added later')
plt.xlim((-5, 5))
plt.ylim((-4, 4))
plt.legend(loc = 'lower left')
plt.show()

pause()

# first plot our pca decision region with LR classifier

lr = LR(multi_class = 'ovr', random_state = 1, solver = 'lbfgs')
lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.xlim((-5, 5))
plt.ylim((-4, 4))
plt.title('Logistic Regression with PCA k = 2 wine data set')
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.show()

pause()

# now use the Scikit-learn PCA tools
# selecting 2 features for visualization

pca = PCA(n_components = 2)
lr2 = LR(multi_class = 'ovr', random_state = 1, solver = 'lbfgs')

X_train_pca2 = pca.fit_transform(X_train_std)
X_test_pca2 = pca.transform(X_test_std)

lr2.fit(X_train_pca2, y_train)
y_pred = lr2.predict(X_test_pca2)

score = ACC_SC(y_test, y_pred)
score2 = lr2.score(X_test_pca2, y_test)

plot_decision_regions(X_train_pca2, y_train, classifier = lr2)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.xlim((-5, 5))
plt.ylim((-4, 4))
plt.title('Logistic Regression with Scikit-Learn PCA k = 2 wine data set')
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.show()                      

pause()















