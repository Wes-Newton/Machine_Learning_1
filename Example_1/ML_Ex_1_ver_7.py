# -*- coding: utf-8 -*-

"""
Conversion of Ex 1 from Andrew Ng's Machine Learning Course
on Coursera from Matlab to Python.  Most but not all of code is from 
johnwittenauer.net - an excellant resource.

My contributions to this example explores the model components a bit more and
compares results to the Normal Equation solution and
Sci-kit Learn Linear Regression Model.

"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import linalg
import sys

# Common functions

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
        # page 5 of example 1 pdf
        Theta_Change = (alpha/len(X)) * (((X*theta.T) - y).T * X)
        theta = theta - Theta_Change
        
        cost[i] = computeCost(X, y, theta)
                  
    return theta, cost

def Compute_Normal_Eqn(X, y):
    b = linalg.inv(X.T * X)
    c = X.T * y
    g = np.dot(b, c)
    return g    

def pause():
    sys.stdout.flush()
    input("Hit Enter to continue")
    
path = os.getcwd() + '\ex1data1.txt'
data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])

print('This is the first set of data - - -')
print()
print("Training Data - DataFrame")
print(data.head()) 
print() 
print("DataFrame Quick Analysis")
print(data.describe())
print()    

path3 = os.getcwd() + '\ex1data2.txt'
data3 = pd.read_csv(path3, header = None, names = ['Square Foot', 'Number of Bedrooms', 'Price'])

print('This is the second set of data - - -')
print()
print("Training Data - DataFrame")
print(data3.head()) 
print() 
print("DataFrame Quick Analysis")
print(data3.describe())
print()    

plt.show()
pause()

print("A  quick plot of the raw data - Population versus Profit")

data.plot(kind = 'scatter', x ='Population', y = 'Profit', figsize = (12,8))
# prepare the first data set
data.insert(0, 'Ones', 1)   # This is the first of training set column Xo = 1!
cols = data.shape[1]        # number of columns (axis = 1)
X = data.iloc[:, 0:cols-1]  # iloc - integer position location
y = data.iloc[:,cols-1:cols]
#X & y is a panda dataframe
Xnp = np.matrix(X.values)
ynp = np.matrix(y.values)
#Xnp & ynp are numpy matrices

print()
print('--------- Gradient Descent ----------')
print()

Xnpcols = len(Xnp.shape)  # this is not correct for all X

theta = np.asmatrix(np.zeros(Xnpcols))

J = computeCost(Xnp, ynp, theta)

alpha = 0.01
iter = 1000   # change 1000 to 10000 yields same results as
              # Sci-Kit learn and Normal Equation at a substantial
              # increase in time to perform calculations .

g, cost = gradientDescent(Xnp, ynp, theta, alpha, iter)
print()
print('This is Linear Regression using the Gradient Descent Method')
print()
print('These are the model coefficients in the form y = b + mx')
print(g)

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
print("This is the Gradient Descent Method for predicting")
print("Profit based upon populations")
print()
print('For a population of 15.0')
print('The model is {} + {} X 15 = {}'.format(g[0,0], g[0,1], g[0,0] + g[0,1] * 15.0))
print()
print('---------------------------------------------------------------------')
print()

plt.show()
pause()

#still using the first data set - subscripts changed to 1 if necessary
print('---------Sci-Kit learn Toolkit----------')
print()

model = linear_model.LinearRegression()
model.fit(Xnp,ynp) #same data as above

x1 = np.array(Xnp[:, 1].A1)
f1 = model.predict(Xnp).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x1, f1, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Sci-Kit learn Predicted Profit vs. Population Size')

g1 = model.coef_
g1_i = model.intercept_[0]

print('These are the Sci-Kit Learn model coefficients in the form y = b + mx')
print(g1_i, g1[0,1])

print()
print("This is the Sci-Kit learn Linear Regression Model for predicting")
print("Profit based upon populations")
print()
print('For a population of 15.0')
print('The model is {} + {} X 15 = {}'.format(g1_i, g1[0,1], g1_i + g1[0,1] * 15.0))
print()
print('By using the predict method of Sci-Kit Learn Linear Regression Model')
print('model.predict([[1,15]] yields {}'.format(model.predict([[1,15]])))
print()
print('---------------------------------------------------------------------')
print()

plt.show()
pause()

#still using the first data set - subscripts changed to 2 if necessary
print("---------------Normal Equation Method---------------- ")
print()

g2 = Compute_Normal_Eqn(Xnp, ynp)

x2 = np.linspace(data.Population.min(), data.Population.max(), 100)
# this is identical to x

# this is the model
g2 = g2.flatten()
f2 = g2[0,0] + (g2[0,1] * x2)  # this creates the predicted line through the data

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x2, f2, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Normal Equation Predicted Profit vs. Population Size')

print()
print("This is the Normal Equation Model for predicting")
print("Profit based upon populations")
print()
print('For a population of 15.0')
print('The model is {} + {} X 15 = {}'.format(g2[0,0], g2[0,1], g2[0,0] + g2[0,1] * 15.0))
print()
print('-------- End of Examples with Data Set 1 - Profit/Population --------')
print()

plt.show()
pause()

#now using the second data data set - subscripts changed to 3 if necessary
print('---------- Gradient Descent on new Data Set - Home Prices -----------')
print()

# prepare the second data set
data3.insert(0, 'Ones', 1)   # This is the first of training set column Xo = 1!
cols3 = data3.shape[1]        # number of columns (axis = 1)
X3 = data3.iloc[:, 0:cols3-1]  # iloc - integer position location
y3 = data3.iloc[:,cols3-1:cols3]
#X & y is a panda dataframe
X3np = np.matrix(X3.values)
y3np = np.matrix(y3.values)
#X3np & y3np are numpy matrices

X3npshape = X3np.shape
X3npcols = X3npshape[1]

theta3 = np.asmatrix(np.zeros(X3npcols))
J3 = computeCost(X3np, y3np, theta3)

alpha3 = .01
iter3 = 25 # change 1000 to 10000 yields same results as
              # Sci-Kit learn and Normal Equation at a substantial
              # increase in time to perform calculations .

g3, cost3 = gradientDescent(X3np, y3np, theta3, alpha3, iter3)
# this crashes
print('These are the model coefficients ')
print(g3)
print()
print('---------------------------------------------------------------------')
print()

plt.show()
pause()

#still using the second data set - subscripts changed to 4 if necessary
#except data is still 3

print('---------Sci-Kit learn Toolkit----------')
print()

print("Sci-Kit learn Linear Regression on new data set - home prices")
print()

model4 = linear_model.LinearRegression(normalize = True)
model4.fit(X3np,y3np)

#Test the model on the training set
f4 = model4.predict(X3np).flatten()
                
g4 = model4.coef_
g4_i = model4.intercept_[0]

print('This is our model and intercept ')
print(g4_i, g4)
print()
#Grab a test case
test4 = X3[5:6]
testnp4 = np.matrix(test4.values)

print('This is our single point test case', testnp4)
print('This yields a predicted sale price of ', model4.predict(testnp4))
print()

#data_length
dl = len(y3np)
y4np = y3np.flatten()  #jump to y4np
diff4 = f4 - y4np  
diff4.resize(dl)
y4np.resize(dl)

x4 = np.arange(dl) #this is a plot index

BR = X3['Number of Bedrooms'] # dataframe for data set 2
br = np.matrix(BR.values)
br.resize(dl)

SF = X3['Square Foot']
sf = np.matrix(SF.values)
sf.resize(dl)

fig_1, axes_1 = plt.subplots(figsize = (12,12), nrows = 3, ncols = 1)
#line chart for actual sale price, projected price by model, and difference
axes_1[0].set_title('Home Pricing Model')    
axes_1[0].set_xlabel('Home Sales')
axes_1[0].set_ylabel('Actual versus predicted and difference')
axes_1[0].plot(x4, y4np, label = 'Actual')   
axes_1[0].plot(x4, f4, label = 'Prediction')
axes_1[0].plot(x4, diff4, label = 'Difference')
axes_1[0].legend(loc=2)
#bar chart for number of bedrooms
axes_1[1].set_xlabel('# of Bedrooms')
axes_1[1].bar(x4, br)  #, label = 'Bedrooms')
#line chart for square feet
axes_1[2].set_xlabel('Square feet')
axes_1[2].plot(x4, sf )     

plt.show()
pause()

print('---------------------------------------------------------------------')
print()

#still using the second data set - subscripts changed to 4 if necessary
#except data is still 3

print("---------------Normal Equation Method---------------- ")
print()

g5 = Compute_Normal_Eqn(X3np, y3np)

f5 = X3np * g5

# diff5 = f5 - y4np  
diff5 = y3np - (X3np * g5)

print('This is our model  ')
components = g5.flatten()
print(components)
print()
print('This is our single point test case', testnp4)
print('This yields a predicted sale price of ',testnp4 * g5)
print()

fig_2, axes_2 = plt.subplots(figsize = (12,12), nrows = 3, ncols = 1)
#line chart for actual sale price, projected price by model, and difference
axes_2[0].set_title('Home Pricing Model')    
axes_2[0].set_xlabel('Home Sales')
axes_2[0].set_ylabel('Actual versus predicted and difference')
axes_2[0].plot(x4, y4np, label = 'Actual')  #reuse y4np and x4
axes_2[0].plot(x4, f5, label = 'Prediction')
axes_2[0].plot(x4, diff5, label = 'Difference')
axes_2[0].legend(loc=2)
#bar chart for number of bedrooms
axes_2[1].set_xlabel('# of Bedrooms')
axes_2[1].bar(x4, br)  #, label = 'Bedrooms')
#line chart for square feet
axes_2[2].set_xlabel('Square feet')
axes_2[2].bar(x4, sf )     # this one is a bar chart

plt.show()
pause()

print('End of Exercise 1')




