# -*- coding: utf-8 -*-
"""

Python Machine Learning Chapter 4
L1 Regularization and Sparse Weights

"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split as T_T_S
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler

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

# LR(penality = 'l1', solver = 'liblinear', multi_class = 'ovr')
# 'lbfgs' - does not support L1
lr = LR(penalty = 'l1', C=1.0, solver= 'liblinear', multi_class = 'ovr')
lr.fit(X_train_std, y_train)
print('Training Accuracy: ', lr.score(X_test_std, y_test))
print()
print('Model ---')
print(lr.intercept_)
# Model ---
# [-1.27130763 -1.46062707 -2.24879006]print()
# intercept for class 1, class 2, & class 3  - theta0 or w0
print(lr.coef_)
# 13 coefficients for each class
fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

weights, params = [], []

for c in np.arange(-4., 6.):
    lr = LR(penalty = 'l1', C=10.**c, solver= 'liblinear', multi_class = 'ovr',
            random_state = 0) # one versus rest
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:,column], label = df_wine.columns[column + 1],
             color = color)
plt.axhline(0, color = 'black', linestyle ='--', linewidth = 3)
plt.xlim([10**-5, 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc = 'upper center', bbox_to_anchor = (1.38, 1.03),
          ncol = 1, fancybox = True)
plt.show()

pause()
   

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    