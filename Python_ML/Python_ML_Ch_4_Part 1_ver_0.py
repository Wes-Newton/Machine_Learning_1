# -*- coding: utf-8 -*-
"""

Python Machine Learning Chapter 4
Missing Data

"""

import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer as SimIM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.compose import ColumnTransformer as COLTR


    
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print('DataFrame with missing data.')
print(df.head())
print()

# returns the number of missing values per column
print('Return the number of missing elements in each column.')
print(df.isnull().sum())
print()

'''
A    0
B    0
C    1
D    1
dtype: int64
'''

# drop rows with missing data
print('The data frame with rows removed with missing data.')
a = df.dropna(axis=0)
print(a)
print()

'''
     A    B    C    D
0  1.0  2.0  3.0  4.0
'''

# drop columns with missing data
print('The data frame with columns removed with missing data.')
b = df.dropna(axis = 1)
print(b)
print()

'''
      A     B
0   1.0   2.0
1   5.0   6.0
2  10.0  11.0
'''

# Or drop only if all are missing
print('The data frame with columns removed with all of the data missing.')
c = df.dropna(how = 'all', axis = 1)
print(c)
print()

'''
      A     B     C    D
0   1.0   2.0   3.0  4.0
1   5.0   6.0   NaN  8.0
2  10.0  11.0  12.0  NaN
'''

# drop rows without 4 real values
print('The data frame with rows removed without 4 real numbers.')
d = df.dropna(thresh=4)
print(d)
print()

'''
     A    B    C    D
0  1.0  2.0  3.0  4.0
'''

# filling data - column average
#e = df.fillna(df.mean()) - is the same as below
e = df.fillna(df.mean(axis = 0), axis = 0)
print('Filling data with the average of the column')
print(e)
print()

'''
      A     B     C    D
0   1.0   2.0   3.0  4.0
1   5.0   6.0   7.5  8.0
2  10.0  11.0  12.0  6.0
'''

# filling data - row average
# e = df.fillna(df.mean(axis = 0), axis = 0)
# NotImplementedError: Currently only can fill with dict/Series column by column
# the above method is not implemented (yet) - manual method below
# https://stackoverflow.com/questions/33058590/pandas-dataframe-replacing-nan-with-row-average

df_copy = df.copy()  # df_copy = df does not create a new copy

m = df.mean(axis=1)
for i, col in enumerate(df):
    df.iloc[:, i] = df.iloc[:, i].fillna(m)

# could not get transpose method to work from SO question    

print('Filling data with the average of the rows')
print(df)    
print()

'''
      A     B          C     D
0   1.0   2.0   3.000000   4.0
1   5.0   6.0   6.333333   8.0
2  10.0  11.0  12.000000  11.0
'''

print('Original data - copied')
print(df_copy)
print()

'''
      A     B     C    D
0   1.0   2.0   3.0  4.0
1   5.0   6.0   NaN  8.0
2  10.0  11.0  12.0  NaN
'''
# sklearn simpleimputer

imr = SimIM(missing_values = np.nan, strategy = 'mean')
imr = imr.fit(df_copy.values) #learns the parameters from the training data
imputed_data = imr.transform(df_copy.values) #using paramters from fit to transform
print('Scikit-Learn SimpleImputer creates a Numpy array')
print(imputed_data)
print()

'''
[[ 1.   2.   3.   4. ]
  [ 5.   6.   7.5  8. ]
  [10.  11.  12.   6. ]]
'''

print('Returning an array with values = True, null values Nan = False')
print(df_copy.isnull())
print()

'''
        A      B      C      D
0  False  False  False  False
1  False  False   True  False
2  False  False  False   True
'''

# Categorical data 

df2 = pd.DataFrame([
         ['green', 'M', 10.1, 'class2'],
         ['red', 'L', 13.5, 'class1'],
         ['blue', 'XL', 15.3, 'class2'],
         ['green','S', 9.1, 'class3']])
df2.columns = ['color', 'size', 'price', 'classlabel']

print('New DataFrame with Categorical Data')
print(df2)
print()

df3 = df2.copy()
df4 = df2.copy()

size_map = {'XL': 4, 'L': 3, 'M': 2, 'S': 1}

df2_copy = df2.copy()
df2['size'] = df2['size'].map(size_map)
print('DataFrame with size mapped to an integer.')
print(df2)
print()

class_map = {label: idx for idx, label in 
                 enumerate(np.unique(df2['classlabel']))}
                 
df2['classlabel'] = df2['classlabel'].map(class_map)    
print('DataFrame with classlabel mapped to an integer class.')
print(df2)
print()

# same process can be used to map backwards to size and classlabel

# using Sci-kit learn LabelEncoder
class_le = LabelEncoder()

# y = class_le.fit_transform(df2['classlabel'].values)

# print(y)
# print()

df3['classlabel'] = class_le.fit_transform(df3['classlabel'])
print('DataFrame with classlabel mapped to an integer class using Scikit-Learn.')
print('LabelEncoder - note size is not mapped.')
print(df3)
print()

print('Scikit-Learn One-hot Encoder')
df4['size'] = df4['size'].map(size_map)
X = df4[['color', 'size', 'price']].values

color_OHE = OHE()
Y = color_OHE.fit_transform(X[:, 0].reshape(-1,1)).toarray()
print(X)
print()
print(Y)

print()
'''
[['green' 'M' 10.1]
 ['red' 'L' 13.5]
 ['blue' 'XL' 15.3]
 ['green' 'S' 9.1]]

[[0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
'''

c_transf = COLTR([('onehot', OHE(), [0]),
                  ('nothing', 'passthrough', [1,2])
                  ])

Y = c_transf.fit_transform(X).astype(float)
print(Y)
print()

'''
[[ 0.   1.   0.   2.  10.1]
 [ 0.   0.   1.   3.  13.5]
 [ 1.   0.   0.   4.  15.3]
 [ 0.   1.   0.   1.   9.1]]
'''

# Get_Dummies method in Pandas

f = pd.get_dummies(df4[['price', 'color', 'size']])
print('Pandas get dummies method.')
print(f)
print()

'''
   price  size  color_blue  color_green  color_red
0   10.1     2           0            1          0
1   13.5     3           0            0          1
2   15.3     4           1            0          0
3    9.1     1           0            1          0
'''

print('Pandas get dummies method - drop_first = True.')
g = pd.get_dummies(df4[['price', 'color', 'size']],
                   drop_first = True)
print(g)
print()

'''
Pandas get dummies method - drop_first = True.
   price  size  color_green  color_red
0   10.1     2            1          0
1   13.5     3            0          1
2   15.3     4            0          0          <---- must be blue
3    9.1     1            1          0
'''



# can be done with One-Hot Encoder
#color_OHE = OHE(categories = 'auto', drop = 'first')
#c_transf = COLTR([('onehot', OHE(), [0]),
#                   ('nothing', 'passthrough', [1,2])
#                   ])

# Y = c_transf.fit_transform(X).astype(float)










