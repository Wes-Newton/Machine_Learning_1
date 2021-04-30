# -*- coding: utf-8 -*-
"""

Python Machine Learning Chapter 4
Sequential Backward Selection


"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as T_T_S
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score as ACC_SC
from sklearn.neighbors import KNeighborsClassifier as KNN

class SBS():
    def __init__(self, estimator, k_features, scoring = ACC_SC,
                 test_size = 0.35, random_state = 1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = T_T_S(X, y, 
                test_size = self.test_size, random_state = self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, 
                                 self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
#                print(p, score)
#                   p =                                    score = 
#                   (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) 0.8863636363636364
#                   (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12) 0.9090909090909091
#                   (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12) 0.9318181818181818

            best = np.argmax(scores)
            self.indices_ = subsets[best]  # finds and saves the best features
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
            #print('Number of dimensions and indices ', dim, self.indices_)
        self.k_score_ = self.scores_[-1]
        return self
        
    def transform(self, X):
        return X[: self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

def pause():
    sys.stdout.flush()
    input("Hit Enter to continue")

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

print('Feature Selection with SBS (sequential backward selection) ')
print()
   
knn = KNN(n_neighbors = 4)
sbs = SBS(knn, k_features = 1)
sbs.fit(X_train_std, y_train)            
            
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker ='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of f\Features')
plt.grid()
plt.tight_layout()

plt.show()
pause()
            
# find the best score and indices of best score          
# if same take lower dimension           

best_index = 0
best_score = 0
 
for i, score in enumerate(sbs.scores_):
    if score >= best_score:
        best_score = score
        best_index = i

best_subset = sbs.subsets_[best_index]
print('The best score is ', best_score)
print('Found with Number of Features = ', len(k_feat) - best_index)
print()
print('Using these columns')
print(best_subset) 
print()
       
best = np.array(best_subset)    
rows = len(X_train_std)  
        
#strip away features that are not selected in the training and test data
X_best_std = np.take(X_train_std, best, axis = 1)       
X_best_test = np.take(X_test_std, best, axis = 1)

# Build a model with the selected features
knn2 = KNN(n_neighbors = 4)
knn2.fit(X_best_std, y_train)       

best_train_score = knn2.score(X_best_std, y_train)  # should be 1.0??
best_test_score = knn2.score(X_best_test, y_test)
        
print('The Training Data with the selected features yield a score of ', 
      best_train_score)
print()
print('Applying the model to the test data yields a score of', best_test_score) 
print()
print('--- NOTE The score for the training data does not match! ---')      
print()

pause()

print('Rerun SBS with the reduced training set')
print()

knn3 = KNN(n_neighbors = 4)
sbs3 = SBS(knn3, k_features = 1) # best = 1.0 matches first run
sbs3.fit(X_best_std, y_train)            
            
k_feat = [len(k) for k in sbs3.subsets_]
plt.plot(k_feat, sbs3.scores_, marker ='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of Features')
plt.grid()
plt.tight_layout()

plt.show()

print('See remarks in code.')
print()

# but if I rerun sbs with the best results the output matches the first run
# I attribute these differences in score methods
# SBS uses sklearn accuracy_score not KNN uses built in score method.

knn3.fit(X_best_std, y_train)

y_pred = knn3.predict(X_best_std)

score3 = ACC_SC(y_train, y_pred)  # , normalize = False) yields 120.

print('The score on the training set with the reduced feature set is ', score3)
print()

# nope still different

