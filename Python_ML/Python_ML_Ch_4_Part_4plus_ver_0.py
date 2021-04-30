# -*- coding: utf-8 -*-
"""

Python Machine Learning Chapter 4
Scikit-Learn Sequential Feature Selection
Forward and Backward

"""
import sys
import pandas as pd
from sklearn.model_selection import train_test_split as T_T_S
from sklearn.ensemble import RandomForestClassifier as RFC
from time import time
from sklearn.feature_selection import SequentialFeatureSelector

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

feat_labels = df_wine.columns[1:]
forest = RFC(n_estimators = 500, random_state = 1)
forest.fit(X_train, y_train)

tic_fwd = time()
sfs_forward = SequentialFeatureSelector(forest, n_features_to_select=5,
                                        direction='forward').fit(X_train,
                                                                 y_train)
toc_fwd = time()

tic_bwd = time()
sfs_backward = SequentialFeatureSelector(forest, n_features_to_select=5,
                                         direction='backward').fit(X_train,
                                                                   y_train)
toc_bwd = time()

print("Features selected by forward sequential selection: "
      f"{feat_labels[sfs_forward.get_support()]}")
print(f"Done in {toc_fwd - tic_fwd:.3f}s")
print("Features selected by backward sequential selection: "
      f"{feat_labels[sfs_backward.get_support()]}")
print(f"Done in {toc_bwd - tic_bwd:.3f}s")

# Out:
# Features selected by forward sequential selection: Index(['Alcohol', 'Ash', 'Magnesium', 'Flavanoids', 'Color Intensity'], dtype='object')
# Done in 254.221s
# Features selected by backward sequential selection: Index(['Malic Acid', 'Magnesium', 'Flavanoids', 'Color Intensity', 'Proline'], dtype='object')
# Done in 352.174s
