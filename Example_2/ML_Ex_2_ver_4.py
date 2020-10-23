# -*- coding: utf-8 -*-

"""
Conversion of Ex 2 from Andrew Ng's Machine Learning Course
on Coursera from Matlab to Python.  Most but not all of code is from 
johnwittenauer.net - an excellant resource.

I worked on plotting the models into a decision map.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys
from sklearn import linear_model

def predict(theta, X):
    probability = sigmoid(X*theta.T)
    return [1 if x>= 0.5 else 0 for x in probability]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta) # passing in np.arrays
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X *theta.T))) # e^Theta.T*X
    second = np.multiply((1-y), np.log(1 - sigmoid(X* theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

def cost(theta, X, y):
    theta = np.matrix(theta) # passing in np.arrays
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X *theta.T)))   
    second = np.multiply((1-y), np.log(1 - sigmoid(X* theta.T)))
    return np.sum(first - second) / (len(X)) 

def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X*theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        if (i==0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = np.sum(term) / len(X) + ((learningRate / len(X)) *theta[:,i])
    return grad

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X*theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    return grad

def pause():
    sys.stdout.flush()
    input("Hit Enter to continue")

# Part 1 - LOGISTIC REGRESSION
print()
print('----------- Logistic Regression - Entrance Exam Scores --------------')
print()

# Take a look at the data
path = os.getcwd() + '\ex2data1.txt'
data = pd.read_csv(path, header = None, names = ['Exam 1', 'Exam 2', 'Admitted'])
print()
print('First Data Set --')
print(data.head())

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize = (12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker ='o', label = 'Accepted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker ='x', label = 'Rejected')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')

plt.show()
pause()

data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(cols - 1)

# Tests
print()
print("Example Validity Tests")
print(cost(theta, X, y))
print(gradient(theta, X, y))
result = opt.fmin_tnc(func = cost, x0 = theta, fprime = gradient, args = (X,y))

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip (predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('Model Accuracy - Logistic Regression = ', accuracy)
print()

plt.show()
pause()

print('Display the decision boundary for Logistic Regression')
print()

# Build a map of the decision boundary
exam1 = np.array([])
for j in range (30, 102, 2):
    index = float(j)
    c = np.full((36), index)
    exam1 = np.hstack((exam1, c))

exam2 = np.array([])
for i in range (0, 36):
    b = np.arange(30., 102., 2. )
    exam2 = np.hstack((exam2, b)) 

# the map dataframe
Xarray = pd.DataFrame({'Ones': 1,'Exam 1': exam1, 'Exam 2': exam2})
Xarraynp = np.array(Xarray.values)

# Run the prediction on the map
map_predictions = predict(theta_min, Xarraynp)

# add the result to the dataframe
Xarray['Result'] = map_predictions

positive = Xarray[Xarray['Result'].isin([1])]
negative = Xarray[Xarray['Result'].isin([0])]

fig2, ax2 = plt.subplots(figsize = (12,8))
ax2.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker ='o', label = 'Likely Accepted')
ax2.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker ='x', label = 'Likely Rejected')
ax2.legend()
ax2.set_xlabel('Exam 1 Score')
ax2.set_ylabel('Exam 2 Score')
ax2.set_title('Prediction Map - Logistic Regression')

plt.show()
pause()

# Regularized Logistic Regression

print('-----Now use Regularized Logistic Regression on the same data set-----')
print()
path3 = os.getcwd() + '\ex2data1.txt'
data3 = pd.read_csv(path3, header = None, names = ['Exam 1', 'Exam 2', 'Admitted'])

degree = 5
learningRate = .3

EX1mean = data3['Exam 1'].mean()
EX1std = data3['Exam 1'].std()
EX2mean = data3['Exam 2'].mean()
EX2std = data3['Exam 2'].std()

data3['Exam 1'] = (data3['Exam 1'] - EX1mean) / EX1std
data3['Exam 2'] = (data3['Exam 2'] - EX2mean ) / EX2std

x1 = data3['Exam 1']
x2 = data3['Exam 2']
data3.insert(3, 'Ones', 1)

for i in range(0, degree):
    for j in range(0, degree):
        data3['F' + str(i) + str(j)] = np.power(x1, i) * np.power(x2,j)

data3.drop('Exam 1', axis = 1, inplace = True)
data3.drop('Exam 2', axis = 1, inplace = True)
data3.drop('F00', axis = 1, inplace = True)

cols3 = data3.shape[1]
theta = np.zeros(cols3 - 1)

cols3 = data3.shape[1]
X3 = data3.iloc[:,1:cols3]
y3 = data3.iloc[:,0:1]        # y at beginning!
X3np = np.array(X3.values)
y3np = np.array(y3.values)

theta3 = np.zeros(cols3-1)
result3 = opt.fmin_tnc(func = costReg, x0 = theta3, fprime = gradientReg, args = (X3np,y3np, learningRate))
theta_min3 = np.matrix(result3[0])

predictions3 = predict(theta_min3, X3np)
correct3 = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip (predictions3, y3np)]
accuracy3 = (sum(map(int, correct3)) % len(correct3))

print('Model Accuracy - Regularized Logistic Regression = ', accuracy3)
print()

# Build a map of the decision boundary
exam1 = np.array([])
for j in range (30, 102, 2):
    index = float(j)
    c = np.full((36), index)
    exam1 = np.hstack((exam1, c))

exam2 = np.array([])
for i in range (0, 36):
    b = np.arange(30., 102., 2. )
    exam2 = np.hstack((exam2, b)) 

# the map dataframe - must normalize data due to the values!
X3map = pd.DataFrame({'Ones': 1,'Exam 1': exam1, 'Exam 2': exam2})
X3map['Exam 1'] = (X3map['Exam 1'] - EX1mean) / EX1std
X3map['Exam 2'] = (X3map['Exam 2'] - EX2mean) / EX2std

x1 = X3map['Exam 1']
x2 = X3map['Exam 2']

for i in range(0, degree):
    for j in range(0, degree):
        X3map['F' + str(i) + str(j)] = np.power(x1, i) * np.power(x2,j)

EX1df  = X3map['Exam 1'] # need copies for plotting
EX2df  = X3map['Exam 2']

X3map.drop('Exam 1', axis = 1, inplace = True)
X3map.drop('Exam 2', axis = 1, inplace = True)
X3map.drop('F00', axis = 1, inplace = True)

X3mapnp = np.array(X3map.values)

map_predictions3 = predict(theta_min3, X3mapnp)

EX1df = (EX1df * EX1std) + EX1mean
EX2df = (EX2df * EX2std) + EX2mean

EX1 = EX1df.values
EX2 = EX2df.values

# The decision map dataframe
Map = pd.DataFrame({'Exam 1': EX1, 'Exam 2': EX2,'Result': map_predictions3})

positive = Map[Map['Result'].isin([1])]
negative = Map[Map['Result'].isin([0])]

fig3, ax3 = plt.subplots(figsize = (12,8))
ax3.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker ='o', label = 'Likely Accepted')
ax3.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker ='x', label = 'Likely Rejected')
ax3.legend()
ax3.set_xlabel('Exam 1 Score')
ax3.set_ylabel('Exam 2 Score')
ax3.set_title('Prediction Map Regularized Logistic Regression')
print('Display the decision boundary for Regularized Logistic Regression')
print()

plt.show()
pause()

print()
print('----------- Logistic Regression - Microchip Testing --------------')
print()

# Part 2 - REGULARIZED LOGISTIC REGRESSION

path2 = os.getcwd() + '\ex2data2.txt'
data2 = pd.read_csv(path2, header = None, names = ['Test 1', 'Test 2', 'Accepted'])

print()
print('First Data Set --')
print(data2.head())

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

plt.show()
pause()


degree = 5
learningRate = 0.0 # try learningRate = 0

x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

# Polynomials from the original features
for i in range(0, degree):
    for j in range(0, degree):
        data2['F' + str(i) + str(j)] = np.power(x1, i) * np.power(x2,j)

data2.drop('Test 1', axis = 1, inplace = True)
data2.drop('Test 2', axis = 1, inplace = True)
data2.drop('F00', axis = 1, inplace = True)


cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1] # y at beginning
X2 = np.array(X2.values)
y2 = np.array(y2.values)

theta2 = np.zeros(cols - 1)

result2 = opt.fmin_tnc(func = costReg, x0 = theta2, fprime = gradientReg, 
                        args = (X2,y2, learningRate))
theta_min = np.matrix(result2[0])

predictions = predict(theta_min, X2)
correct = [1 if ((a ==1 and b==1) or (a ==0 and b==0)) else 0 for (a,b) in zip(predictions, y2)]
A = sum(map(int, correct))
B = len(correct)

Accuracy = (A / float( B)) * 100
Acc = str(Accuracy)

print('Model Accuracy - Regularized Logistic Regression = ', Accuracy)
print()

# Accept Contour Visualization
X_predict = np.linspace(1.0, 1.0, degree**2)

predict2 = np.array([0.0, 0.0, 1])

for xx1change in range(0, 251):
    xx1 = (xx1change -100) / 100.
    for xx2change in range(0, 251):
        xx2 = (xx2change - 100) / 100.
        idx = 0
        for i in range(0, degree):
            for j in range(0, degree):
                X_predict[idx] = float((xx1**i) * xx2**j)
                idx += 1
        output = (sigmoid(X_predict * theta_min.T))
        if output >= 0.5:
            Accept = 1
        else:
            Accept = 0
        add = np.array([xx1, xx2, Accept])    
        predict2 = np.vstack((predict2, add))
     
data4 = pd.DataFrame(predict2) 
positive2 = data4[data4[2].isin([1])]
fig2, ax2 = plt.subplots(figsize = (12,8))
ax2.scatter(positive2[0], positive2[1], s=50, c='r', marker ='x', label = 'Accept')

plt.xlim(-1, 1.5)
plt.ylim(-1, 1.5)
ax2.set_xlabel('Test 1 Score')
ax2.set_ylabel('Test 2 Score')
chart_title = 'Prediction Map, Degree = ' + str(degree) + ', Learning Rate = ' + \
            str(learningRate) + ', Accuracy = ' + Acc[:4]
ax2.set_title(chart_title)
print('Display the decision boundary for Regularized Logistic Regression')
print()

plt.show()
pause()

print()
print('------- Sci-Kit Learn Logistic Regression - Microchip Testing --------')
print()

model = linear_model.LogisticRegression(penalty='l2', C=1.0, max_iter = 1000)
model.fit(X2, y2.ravel())
score =  model.score(X2, y2)
print('Model coefficients are :')
print(model.coef_)
print('Model Intercept = ', model.intercept_)
print()
print('This tool yields a model accuracy of = {0:0.2f}'.format(score))
print()
print('----- End of Example 2 -----')
print()



