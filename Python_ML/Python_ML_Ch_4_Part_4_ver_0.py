# -*- coding: utf-8 -*-
"""

Python Machine Learning Chapter 4
Random Forest feature Selection

"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as T_T_S
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score as ACC_SC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectFromModel as SFM

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

feat_labels = df_wine.columns[1:]

forest = RFC(n_estimators = 500, random_state = 1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]],
                                     importances[indices[f]]))
    # 1) Proline                        0.213565
                       
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align = 'center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices] , rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

pause()

sfm = SFM(forest, threshold = 0.1, prefit = True)
X_train_best2 = sfm.transform(X_train)
X_test_best2 = sfm.transform(X_test)

forest_best = RFC(n_estimators = 1000, random_state = 1)
forest_best.fit(X_train_best2, y_train)
y_pred = forest_best.predict(X_train_best2) 

score_train = ACC_SC(y_train, y_pred) 
print('With the top 5 features on the training set the score is ', score_train)
print()
y_pred2 = forest_best.predict(X_test_best2)
score_test = ACC_SC(y_test, y_pred2)
print('With the top 5 features on the test set the score is ', score_test)
print()

print('--- End of Chapter 4 ---')




