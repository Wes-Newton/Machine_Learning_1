# -*- coding: utf-8 -*-

"""
Conversion of Ex 6 from Andrew Ng's Machine Learning Course
on Coursera from Matlab to Python.  This is from: 

    johnwittenauer.net.

He implemented this with the Sci-Kit learn SVM implementation.

I added a few twists including:
    
A modified Decision Boundary Routine from 'Python Machine Learning' added.

Notes:
    Clear variables before each run
#https://stackoverflow.com/questions/45853595/spyder-clear-variable-explorer
#    -along-with-variables-from-memory
"""


from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
import sys
from sklearn import svm

#%matplotlib inline
#https://stackoverflow.com/questions/29356269/plot-inline-or-a-separate-window-using-matplotlib-in-spyder-ide
#https://stackoverflow.com/questions/30878666/matplotlib-python-inline-on-off

def pause():
    sys.stdout.flush()
    input("Hit Enter to continue")

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02,
                          label = None, range = None):
    '''
    This routine is from the book - Python Machine learning
    
    label and range added by me for this routine.  Really nice
    routine for decision boundaries.

    Parameters
    ----------
    X : TYPE 
        DESCRIPTION. Test data 2D
    y : TYPE
        DESCRIPTION. Classification
    classifier : TYPE
        DESCRIPTION. Sci-Kit Learn Object
    test_idx : TYPE, optional
        DESCRIPTION. For splitting the data into training and test data.
    resolution : TYPE, optional
        DESCRIPTION. Meshgrid resolution, the default is 0.02.
    label : TYPE, optional
        DESCRIPTION. Chart Title

    Returns
    -------
    None.

    '''
    if range == None:
        range = 1
    markers = ['s', 'x', 'o', '^', 'v']
    colors = ['r', 'b', 'g', 'm', 'c']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:,0].min() - range, X[:, 0].max() + range  
    x2_min, x2_max = X[:,1].min() - range, X[:, 1].max() + range
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
    if label != None:
        plt.title(label)
    plt.show()

#-----------------------------------------------------------------------------

print('---- Part 1. SVM on simple 2D Data Set ----')
print()

raw_data = loadmat('ex6data1.mat')
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']
print('Sample of Data')
print(data.head())
print('....')
print(data.tail())
print()
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]
X = np.array(data[['X1', 'X2']])
y = data['y']

#Plot of raw data
fig1, ax1 = plt.subplots(figsize=(12,8))
ax1.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax1.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax1.set_xlabel('X1 Data')
ax1.set_xlabel('X2 Data')
ax1.set_title('2D data set for SVM classification')
ax1.legend(loc = 'lower left')

plt.show()
pause()

print('---- SVM C = 1 ----')
print()
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
# can be alternatively implemented as svm = SGDClassifier(loss = 'hinge')
svc.fit(data[['X1', 'X2']], data['y'])
svc_score1 = svc.score(data[['X1', 'X2']], data['y'])
data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])
fig2, ax2 = plt.subplots(figsize=(12,8))
ax2.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], 
            cmap='seismic')
ax1.set_xlabel('X1 Data')
ax2.set_xlabel('X2 Data')
ax2.set_title('SVM (C=1) Decision Confidence')

plt.show()
pause()

# Decision Regions C=1
title = 'SVM C = 1'
plot_decision_regions(X, y, classifier = svc, label = title)
print('The percentage accuracy for C = 1 is ','{:.2%}'.format(svc_score1))
if svc_score1 == 1.0:
    print('Outlier Captured.')
else:
    print('Fails to capture outlier ..')
print()

pause()

print('---- SVM C = 90 ----')
print()
svc2 = svm.LinearSVC(C=95, loss='hinge', max_iter=5000)
svc2.fit(data[['X1', 'X2']], data['y'])
svc_score2 = svc2.score(data[['X1', 'X2']], data['y'])
data['SVM 2 Confidence'] = svc2.decision_function(data[['X1', 'X2']])
fig3, ax3 = plt.subplots(figsize=(12,8))
ax3.scatter(data['X1'], data['X2'], s=50, c=data['SVM 2 Confidence'], 
            cmap='seismic')
ax3.set_xlabel('X1 Data')
ax3.set_xlabel('X2 Data')
ax3.set_title('SVM (C=95) Decision Confidence')

plt.show()
pause()

# Decision Regions C=90
title = 'SVM C = 95'
plot_decision_regions(X, y, classifier = svc2, label = title)
print('The percentage accuracy for C = 95 is ','{:.2%}'.format(svc_score2))
if svc_score2 == 1.0:
    print('Outlier Captured.')
else:
    print('Fails to capture outlier ..')
print()

pause()

print(data.head())
print()

#https://scikit-learn.org/stable/modules/svm.html#svm-mathematical-formulation

print('---- Part 1 Continued. SVC on non-linear 2D Data Set ----')
print()

raw_data2 = loadmat('ex6data2.mat')

data2 = pd.DataFrame(raw_data2['X'], columns=['X1', 'X2'])
data2['y'] = raw_data2['y']

positive = data2[data2['y'].isin([1])]
negative = data2[data2['y'].isin([0])]

print('Sample of Data')
print(data2.head())
print('....')
print(data2.tail())
print()


fig4, ax4 = plt.subplots(figsize=(6,6))
ax4.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
ax4.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
ax4.set_xlabel('X1 Data')
ax4.set_ylabel('X2 Data')
ax4.set_title('2D data set for SVC non-linear classification')
ax4.legend(loc = 'upper right')

plt.show()
pause()

X2 = np.array(data2[['X1', 'X2']])
y2 = data2['y']

svc3 = svm.SVC(C=100, gamma=10, probability=True, degree = 3)
# default kernel - 'rbf'
svc3.fit(data2[['X1', 'X2']], data2['y'])
svc_score3 = svc3.score(data2[['X1', 'X2']], data2['y'])
data2['Probability'] = svc3.predict_proba(data2[['X1', 'X2']])[:,0]

fig5, ax5 = plt.subplots(figsize=(12,8))
ax5.scatter(data2['X1'], data2['X2'], s=30, c=data2['Probability'], cmap='Reds')
ax5.set_xlabel('X1 Data')
ax5.set_xlabel('X2 Data')
ax5.set_title('2D data set for SVC non-linear classification')

plt.show()
pause()

# Decision Regions C=100
title = 'SVC classifier'
plot_decision_regions(X2, y2, classifier = svc3, label = title,
                      resolution = .001, range = 0.0)
print('The percentage accuracy for C = 100 is ','{:.2%}'.format(svc_score3))
print()

print("Now look at some of the attributes of the SVC model - svc3")
print()
print('svc3 classes are ', svc3.classes_)
print('svc3 dual coefficients are ', svc3.dual_coef_)
print('svc3 fit status = ', svc3.fit_status_)
print('svc3 intercept = ', svc3.intercept_)
print('The number of support vectors = ', len(svc3.support_))
print()

pause()

raw_data3 = loadmat('ex6data3.mat')
X = raw_data3['X']
Xval = raw_data3['Xval']
y = raw_data3['y'].ravel()
yval = raw_data3['yval'].ravel()
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
best_score = 0
best_params = {'C': None, 'gamma': None}
scores = []

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

print('The best score is ','{:.2f}'.format(best_score))
print('From the parameters', best_params)
print()

# See color chart of parameters at:
#h ttps://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html


# Skipped Part 2 Spam Filter for now.