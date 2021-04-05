# -*- coding: utf-8 -*-
"""
Chapter 3 Examples from Python machine Learning

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

# Sci-Kit Learn Notes:
# Perceptron is a classification algorithm which shares the same underlying 
# implementation with SGDClassifier. In fact, Perceptron() is equivalent to 
# SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", 
# penalty=None).

class LogisticRegressionGD(object):
    
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta=eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01,
                              size = 1 + X.shape[1])
        self.cost_ = []
        
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)           
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            #sigmoid implementation for logistic regression
            cost = (-y.dot(np.log(output) - (1 -y).dot(np.log(1 -output))))
            
            #cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def activation(self, z):
        # compute logistic sigmoid activation
        sigmoid = 1. / (1. + np.exp(-np.clip(z, -250, 250)))
        
        return sigmoid
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.5, 1, 0)

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
print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                    random_state = 1, stratify = y)
# train_test_split - randomly split the data set into train and test
# samples with 30% going into test size (45) and 70% into training
# samples (105).
# random_state = 1 uses a constant seed so the results are reproducible
# stratify = y ensures the class labels are equally represented.

print('Data Set:')
print('X size = ', len(X))
print('X test size =', len(X_test))
print('y size = ', np.bincount(y))
print('y test size = ', np.bincount(y_test))
print()
print('---- Sci-kit Learn Perceptron ----')
print()
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0 = 0.1, random_state = 1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
misclassified = (y_test != y_pred).sum()
print('Number misclassified = ', misclassified)
percent_correct = acc_sc(y_test, y_pred)
#print('Accuracy =  %.3f' % percent_correct)
print('The percentage accuracy is ','{:.2%}'.format(percent_correct))
print()

X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined,y=y_combined, classifier = ppn,
                      test_idx = range(105, 150))
plt.xlabel('Petal Length [standardized]')
plt.ylabel('sepal length [standardized]')
plt.title('Sci-Kit Learn - Perceptron')
plt.legend(loc = 'upper left')
plt.tight_layout()

plt.show()

pause()

# Logistic Regression
# this implementation is binary class (1,0 - yes, no, -...)
print()
print('---- Logistic Regression with Gradient Descent ----')
print()

X_train_LR = X_train_std[(y_train == 0) | (y_train == 1)]
X_test_LR = X_test_std[(y_test == 0) | (y_test == 1)]
y_train_LR = y_train[(y_train == 0) | (y_train == 1)]
y_test_LR = y_test[(y_test == 0) | (y_test == 1)]

lrgd = LogisticRegressionGD(eta = 0.05, n_iter = 1000, random_state = 1)
                            
lrgd.fit(X_train_LR, y_train_LR)

y_pred2 = lrgd.predict(X_test_LR)
misclassified2 = (y_test_LR != y_pred2).sum()
print('Misclassified to test = ', misclassified2)

plot_decision_regions(X = X_train_LR, y = y_train_LR, classifier = lrgd)
plt.xlabel('petal length [standardized]')                     
plt.ylabel('sepal length [standardized]')
plt.title('Logistic Regression with Gradient Descent')
plt.legend(loc = 'upper left')
plt.tight_layout()

plt.show()
pause()

# Logistic Regression - Sci-kit learn
# this implementation is multi-class
print()
print('---- Sci-kit Learn Multi-Class Logistic Regression ----')
print()

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs',
                        multi_class= 'multinomial')
# see notes below 'lbgfs' is default solver for v0.22
# limited memory Broyden-Fletcher-Goldfarb-Shanno
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined, y_combined, classifier = lr,
                      test_idx = range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('sepal length [standardized]')
plt.title('Sci-Kit Learn Logistic Regression (solver = lbfgs)')
plt.legend(loc = 'upper left')
plt.tight_layout()

y_pred3 = lr.predict(X_test_std)
misclassified3 = (y_test != y_pred3).sum()
print('Misclassified to test = ', misclassified3)
print('Score Method')
score = lr.score(X_test_std, y_test)
print('The percentage accuracy is ','{:.2%}'.format(score))
print()
print('first five probabilities of test set')
print(lr.predict_proba(X_test_std[:5, :]))
print('Actual - ', y_test[:5], ',   Predicted - ', 
      lr.predict_proba(X_test_std[:5, :]).argmax(axis=1))
print()

plt.show()
pause()


# Sci-Kit Learn Notes:
#
# solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
# Algorithm to use in the optimization problem.
# For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ 
# are faster for large ones.
# For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle 
# multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
# ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
# ‘liblinear’ and ‘saga’ also handle L1 penalty
# ‘saga’ also supports ‘elasticnet’ penalty
# ‘liblinear’ does not support setting penalty='none'
# Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features 
# with approximately the same scale. You can preprocess the data with a scaler
# from sklearn.preprocessing.
# New in version 0.17: Stochastic Average Gradient descent solver.
# New in version 0.19: SAGA solver.
# Changed in version 0.22: The default solver changed from ‘liblinear’ to 
# ‘lbfgs’ in 0.22.
# L-BFGS-B – Software for Large-scale Bound-constrained Optimization
# Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales. 
# http://users.iems.northwestern.edu/~nocedal/lbfgsb.html                

print()
print('---- Sci-kit Learn Support Vector Machine ----')
print()

svm = SVC(kernel = 'linear', C = 1.0, random_state =1)
svm.fit(X_train_std, y_train)

#X_combined is standardized
plot_decision_regions(X_combined,y_combined, 
                      classifier = svm, test_idx = range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('sepal length [standardized]')
plt.title('Sci-Kit Learn Support Vector Machine - C = 1.0')
plt.legend(loc = 'upper left')
plt.tight_layout()

plt.show()
pause()

svm2 = SVC(kernel = 'linear', C = 100.0, random_state =1)
svm2.fit(X_train_std, y_train)

#X_combined is standardized
plot_decision_regions(X_combined,y_combined, 
                      classifier = svm2, test_idx = range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('sepal length [standardized]')
plt.title('Sci-Kit Learn Support Vector Machine - C = 100.0')
plt.legend(loc = 'upper left')
plt.tight_layout()

plt.show()
pause()

# End of Chapter 3 Examples - Part 1.

    
