# -*- coding: utf-8 -*-
"""

This was not covered at johnwittenauer.net (that I found).  The cost function and 
gradient are found in the previous examples.

I compared the Linear Regression that is computed 
(with cost function that is minimized with 'minimize' and 'fmin_tnc')
to Linear Regression for Sci-kit Learn and the solution to the Normal
Equation.

The 8th order polynomial matches that from the class.

Scikit-learn Polynomial Regression added.

"""

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.optimize import fmin_tnc
import sys
from sklearn.linear_model import LinearRegression
from scipy import linalg

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

#from sklearn.multiclass import OutputCodeClassifier
#from sklearn.svm import LinearSVC
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.neural_network import MLPClassifier
#import os
#import pandas as pd
#from sklearn import linear_model
#from sklearn.preprocessing import OneHotEncoder
#import random

def pause():
    sys.stdout.flush()
    input("Hit Enter to continue")

def computeCost(theta, X, y, learning_rate):
    m = len(y)
    inner = np.square((X * theta.T) - y)
    J = np.sum(inner) / (2 * m)

    # Do not regularize theta0 term
    lambda_change = learning_rate / (2*m) * (theta[1:].T * theta[1:])
    J = J + lambda_change
    return J[0]      # [0] is a hack, not sure why it is needed

def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = (X*theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        if (i==0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = np.sum(term) / len(X) + ((learningRate / len(X)) *theta[:,i])
    return grad

def trainLinearReg(X, y, learning_rate):
    m, n = np.shape(X)
    theta = np.ones(n)  #changed to ones from zeros makes solutions all the same.
    fmin = minimize(fun=computeCost, x0=theta, args=(X, y, learning_rate), method='TNC', jac=gradientReg)
    result = fmin_tnc(func = computeCost, x0 = theta, fprime = gradientReg, args = (X,y, learning_rate))
    all_theta = fmin.x
    return all_theta, result

def Compute_Normal_Eqn(X, y):
    b = linalg.inv(X.T * X)
    c = X.T * y
    g = np.dot(b, c)
    return g    

def addPoly(X, iter):
    power=2
    rows=X.shape[0]
    for columns in range(1, iter):
        Xnewcol = np.ones(rows)
        for i in range(rows):
            Xnewcol[i] = X[i,1]**power
        X = np.insert(X,power,Xnewcol,axis=1)
        power += 1
    return X    

def featureNormalization(X):
    X1 = X.copy()
    columns=X.shape[1]
    row=X.shape[0]  
    #do for all columns
    #return the mean and std for later
    mean = []
    std = []
    mean.append(1)
    std.append(0.0)
    for column in range(1,columns):
        mean.append(np.mean(X[column]))
        std.append(np.std(X[column]))
        #now modify elements
        for row in range(0, rows):
            X1[row,column] = (X[row,column]- mean[column]) / std[column]
    return X1, mean, std

# Load and plot data
print(' ---------- Example 5 --------------')
print()

data = loadmat('ex5data1.mat')
X = data['X']
X1 = data['X'] #Use for plots
y = data['y']
rows = data['X'].shape[0]
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

print('---- This is a plot of the data -----')
print()

fig1, ax1 = plt.subplots(figsize = (12,8))
ax1.scatter(X1, y, color = 'red')
ax1.set_xlabel('Change in Water Level')
ax1.set_ylabel('Water Flowing Out of Dam')
ax1.set_title('Training Data')
#https://stackoverflow.com/questions/56115201/typeerror-ufunc-sqrt-not-
#supported-for-the-input-types-when-plotting-a-colorm

plt.show()
pause()

# Regularized Linear Regression
print('---- Regularized Linear Regression ----')
print('Optimizing using minimize and method="tnc" and fmin_tnc ')
print()

learning_rate = 0 #lambda
theta, result = trainLinearReg(X, y, learning_rate)

print('Theta for minimize = ', theta)
print('Theta for fmin_tnc = ', result)
print('28 is number of evaluations and 4 is a return code Linear Search Failed')
print()
print('Now plot the first order estimate')
print()

ypred = theta[0] + X1*theta[1]  # = theta*X

fig2, ax2 = plt.subplots(figsize = (12,8))
ax2.plot(X1, ypred, color='green')
ax2.scatter(X1, y, color = 'red')
ax2.set_xlabel('Change in Water Level')
ax2.set_ylabel('Water Flowing Out of Dam')
ax2.set_title(' First Order Estimate ')

plt.show()
pause()

# Sci-Kit learn linear regression
print('---- Sci-Kit Learn Linear Regression Tool ---- ')
print()
reg = LinearRegression().fit(X,y)
a = reg.score(X,y)
b = reg.coef_
c=reg.intercept_
d = reg.predict(X)
print('The SciKit Learn intercept is ', c)
print('The coefficients is/are ', b)

print('Now plot the Sci-Kit learn Linear Regression Prediction')
print()

fig3, ax3 = plt.subplots(figsize = (12,8))
ax3.plot(X1, d, color='black')
ax3.scatter(X1, y, color = 'red')
ax3.set_xlabel('Change in Water Level')
ax3.set_ylabel('Water Flowing Out of Dam')
ax3.set_title(' Sci-Kit Learn Linear Regression ')

plt.show()
pause()

# Normal Equation Method
print('---- Normal Equation Method ---- ')
print()

m = X.shape[0]
Xn = np.matrix(X)
yn = np.matrix(y)

theta = Compute_Normal_Eqn(Xn,yn)
print('The normal equation intercept is ', theta[0])
print('And coeeficient is ', theta[1])

ypred2 = X * theta

print('Now plot the Normal Equation Solution ')
print()

fig4, ax4 = plt.subplots(figsize = (12,8))
ax4.plot(X1, ypred2, color='blue')
ax4.scatter(X1, y, color = 'red')
ax4.set_xlabel('Change in Water Level')
ax4.set_ylabel('Water Flowing Out of Dam')
ax4.set_title(' Normal Equation Method ')

plt.show()
pause()

# Learning Curve for Linear Regression
print('---- Learning Curve for Linear Regression ---- ')
print()

error_train =[]
error_val =[]
Xval = data['Xval']
yval = data['yval']
rows2 = data['Xval'].shape[0]
Xval = np.insert(data['Xval'], 0, values=np.ones(rows2), axis=1)
thetaval = np.array([1,1])

learning_rate = 0

theta1, result = trainLinearReg(X, y, learning_rate)
Jstd = computeCost(theta1, Xval, yval, learning_rate)

for i in range(1,13):
    Xtrain = X[0:i]
    ytrain = y[0:i]
    thetai, result = trainLinearReg(Xtrain, ytrain, learning_rate)
    Jtrain = computeCost(thetai, Xtrain, ytrain, learning_rate)
    error_train.append(Jtrain)
    Jval = computeCost(thetai, Xval, yval, learning_rate)
    error_val.append(Jval)

j = [i for i in range(12)]

fig5, ax5 = plt.subplots(figsize = (12,8))
ax5.plot(j, error_train, color='blue')
ax5.plot(j, error_val, color = 'red')
ax5.set_xlabel('Number of Training SamplesChange in Water Level')
ax5.set_ylabel('Error')
ax5.set_title('Learning Curve for Linear Regression ')

plt.show()
pause()

# Polynomial Regression
print('---- Polynominal Regression Order = 8 ---- ')
print()

Xnormal, mean, std = featureNormalization(X)
order = 8 
Xpoly = addPoly(Xnormal, order)
   
learning_rate = 0 #lambda
thetapoly, result = trainLinearReg(Xpoly, y, learning_rate)

print('Theta for minimize = ', theta)
print('Theta for fmin_tnc = ', result)
print('Number of evaluations and return code ')
print()
print('Now plot the eighth order estimate')
print()

ypred2 = np.dot(Xpoly, thetapoly)

fig6, ax6 = plt.subplots(figsize = (12,8))
ax6.scatter(X1, ypred2, color='green')
ax6.scatter(X1, y, color = 'red')
ax6.set_xlabel('Change in Water Level')
ax6.set_ylabel('Water Flowing Out of Dam')
ax6.set_title('Polynomial Order = 8 ')

plt.show()
pause()       

# Learning Curve for Polynomial Regression
print('---- Learning Curve for Polynominal Regression ---- ')
print()
        
Jstd = computeCost(thetapoly, Xpoly, y, learning_rate)        

Xvalpoly = addPoly(Xval, order)

error_train2 =[]
error_val2 =[]

# this can be placed in a loop with learning_rate(lambda) as the iterator as well
# skipped for now.
        
for i in range(1,13):
    Xtrain = Xpoly[0:i]
    ytrain = y[0:i]
    thetai, result = trainLinearReg(Xtrain, ytrain, learning_rate)
    Jtrain = computeCost(thetai, Xtrain, ytrain, learning_rate)
    error_train2.append(Jtrain)
    Jval = computeCost(thetai, Xvalpoly, yval, learning_rate)
    error_val2.append(Jval)        
        
fig7, ax7 = plt.subplots(figsize = (12,8))
ax7.scatter(j, error_train2, color='green')
ax7.scatter(j, error_val2, color = 'red')
ax7.set_xlabel('Number of Training Samples')
ax7.set_ylabel('Error')
ax7.set_title('Learning Curve for Polynomial Regression ')

plt.show()
pause()           
       
# Polynomial Regression with Sci-Kit Learn
print('---- Polynominal Regression with Sci-Kit Learn Order = 8 ---- ')
print()

model = make_pipeline(PolynomialFeatures(8), Ridge())
model.fit(Xnormal, y)

y_plot = model.predict(Xnormal)

fig8, ax8 = plt.subplots(figsize = (12,8))
ax8.scatter(X1, y_plot, color='blue')
ax8.scatter(X1, y, color = 'red')
ax8.set_xlabel('Change in Water Level')
ax8.set_ylabel('Water Flowing Out of Dam')
ax8.set_title('Scikit-Learn PolynomialRegression Order = 8 ')

plt.show()
pause()           

# Polynomial Regression with Sci-Kit Learn
print('---- Polynominal Regression with Sci-Kit Learn Order 2 through 8 ---- ')
print()

#https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_
#interpolation.html#sphx-glr-auto-examples-linear-model-plot-polynomial-interpolation-py

x_fit = np.array([np.ones(61), np.linspace(-2.5, 3.5, 61)])
x_fit = x_fit.T
x_line = np.linspace(-2.5, 3.5, 61)
x2_line = (x_line * std[1]) + mean[1]

colors = ['b', 'g', 'r', 'c', 'm', 'y','k']
fig10, ax10 = plt.subplots(figsize = (12, 8))

for count, degree in enumerate([2, 3, 4, 5, 6, 7, 8]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(Xnormal, y)    
    y_line = model.predict(x_fit)
    ax10.plot(x2_line, y_line, color = colors[count], label = "Degree = %d" % degree)

ax10.set_xlabel('Change in Water Level')
ax10.set_ylabel('Water Flowing Out of Dam')
ax10.set_title('Scikit-Learn PolynomialRegression Order = 2 through 8 ')
ax10.scatter(X1, y, color = 'red')
plt.legend(loc = 'upper left')

plt.show()
pause()

# End of Exercise 5    