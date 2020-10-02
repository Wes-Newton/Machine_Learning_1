# -*- coding: utf-8 -*-
"""

Conversion of Ex 1 from Andrew Ng Machine Learning Course
on Coursera.  Most but not all of code is from 
johnwittenauer.net - an excellant resource.

This example explores the model components a bit more.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# exercise #1 - Andrew Ng Coursera Machine Learning Course
# Most of this code is from johnwittenauer.net

path = os.getcwd() + '\ex1data1.txt'

data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])

print()
print("Training Data - DataFrame")
print(data.head()) 
print() 
print("DataFrame Quick Analysis")
print(data.describe())
print()

data.plot(kind = 'scatter', x ='Population', y = 'Profit', figsize = (12,8))

def computeCost(X, y, theta):
    inner = np.square((X * theta.T) - y)
    J = np.sum(inner) / (2 * len(X))
    return J
    
def gradientDescent(X, y, theta, alpha, iters):
    
    # from johnwittenauer.net
    # temp = np.matrix(np.zeros(theta.shape))
    # parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        # #from johnwittenauer.net
        # getting incorrect result from this
        # error = (X * theta.T) - y
        # for j in range(parameters):
        #     term = np.multiply(error, X[:,j])
        #     temp[0,j] = theta[0, j] = ((alpha / len(X)) * np.sum(term))
        # theta = temp
        # cost[i] = computeCost(X, y, theta)
        
        # Mine from course 

        Theta_Change = (alpha/len(X)) * (((X*theta.T) - y).T * X)
        theta = theta - Theta_Change
        cost[i] = computeCost(X, y, theta)
           
    return theta, cost

data.insert(0, 'Ones', 1)   # This is the first of training set column Xo = 1!
cols = data.shape[1]        # number of columns (axis = 1)

X = data.iloc[:, 0:cols-1]  # iloc - integer position location
        #.iloc[: 0:3-1]
y = data.iloc[:,cols-1:cols]
        #.iloc[:2:3]
#X & y is a panda dataframe

X = np.matrix(X.values)
y = np.matrix(y.values)
#X & y is noW a numpy matrix

theta = np.matrix(np.array([0,0]))
J = computeCost(X, y, theta)

alpha = 0.01
iter = 1000
g, cost = gradientDescent(X, y, theta, alpha, iter)

print('These are the model coefficients in the form y = b + mx')
print(g)

# This generates the model
# g is the linear coefficients for the line f = g[0,0] + g[0,1]*x
x = np.linspace(data.Population.min(), data.Population.max(), 100)

# this is the model
f = g[0,0] + (g[0,1] * x)

fig, ax = plt.subplots(figsize = (12,8))
ax.plot(x, f, 'r', label = 'Prediction')
ax.scatter(data.Population, data.Profit, label = 'Training Data')
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

fig1, ax1 = plt.subplots(figsize = (12,8))
ax1.plot(np.arange(iter), cost, 'r')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Cost')
ax1.set_title('Error vs. Training Epoch')

print()
print("This is the Linear Regression Model for predicting")
print("Profit based upon populations")
print()
print('For a population of 15.0')
print('The model is {} + {} X 15 = {}'.format(g[0,0], g[0,1], g[0,0] + g[0,1] * 15.0))
print()
print()
print('----------------------------------------')
print()
print()

print('---------Sci-Kit learn Toolkit----------')

model = linear_model.LinearRegression()
model.fit(X,y)
x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()
                  

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Sci-Kit learn Predicted Profit vs. Population Size')

g2 = model.coef_
g2_i = model.intercept_[0]

print('These are the Sci-Kit Learn model coefficients in the form y = b + mx')
print(g2_i, g2[0,1])

print()
print("This is the Sci-Kit learn Linear Regression Model for predicting")
print("Profit based upon populations")
print()
print('For a population of 15.0')
print('The model is {} + {} X 15 = {}'.format(g2_i, g2[0,1], g2_i + g2[0,1] * 15.0))
print()
print('By using the predict method of Sci-Kit Learn Linear Regression Model')
print('model.predict([[1,15]] yields {}'.format(model.predict([[1,15]])))
print()
print('End of Exercise 1')

















