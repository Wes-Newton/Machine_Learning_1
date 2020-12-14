# -*- coding: utf-8 -*-


"""
Conversion of Ex 3 from Andrew Ng's Machine Learning Course
on Coursera from Matlab to Python.  Most but not all of code is from 
johnwittenauer.net.
I could not get the cost function from his site to match the results
from the class.  My version did but I cannot achieve the desired
accuracy or match the results acheived at johnwittenauer.net

I created the plotting routine for matplotlib of the sample set
and mis-classified samples.

Added two multi-class classification methods from Sci-Kit Learn

"""


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import sys

from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    
    return np.array(grad).ravel()

def cost(theta, X, y, learningRate):
 
    theta = np.matrix(theta) # passing in np.arrays
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(X)
    # y = 1 case
    first = np.sum(-y * np.log(sigmoid(X*theta.T)))
    # y = 0 case
    second = np.sum((1-y) * np.log(1 - sigmoid(X*theta.T)))
    reg = (learningRate / (2 * m)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    J = ((first - second) / m) + reg

    return J

def cost2(theta, X, y, learningRate):
    # this is similar to the one above but the extra '.T' aligns the vectors
    theta = np.matrix(theta) # passing in np.arrays
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(X)
    # y = 1 case
    first = np.sum(-y * np.log(sigmoid(X*theta.T).T))
    # y = 0 case
    second = np.sum((1-y) * np.log(1 - sigmoid(X*theta.T).T))
    reg = (learningRate / (2 * m)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    J = ((first - second) / m) + reg
    
    return J

def costJWN(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg
    
def predict(theta, X):
    probability = sigmoid(X*theta.T)
    return [1 if x>= 0.5 else 0 for x in probability]
    
def costReg(theta, X, y, learningRate):
    # returns correct gradients for sample data
    z = X.dot(theta)
    h = sigmoid(z)
    theta = np.matrix(theta) # passing in np.arrays
    Xt = np.matrix(X)
    yt = np.matrix(y)
    m = len(X)
    # y = 1 case   
    first = np.sum(-yt * np.log(sigmoid(Xt*theta.T)))   #e^Theta.T*X
    # y = 0 case
    second = np.sum((1-yt) * np.log(1 - sigmoid(Xt*theta.T)))
    #same result with below
    # first = np.multiply(-yt, np.log(sigmoid(Xt.dot(thetat.T)) ))  #e^Theta.T*X
    # second = np.multiply((1-yt), np.log(1 - sigmoid(Xt.dot(thetat.T))))
    reg = (learningRate / (2 * m)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    # or this way
    #reg = (learningRate / (2 * m)) * np.sum(thetat[:,1:thetat.shape[1]].T*thetat[:,1:thetat.shape[1]])
    J = ((first - second) / m ) + reg
    length = theta.shape
    grad = np.zeros(length[1])
    grad = (1 / m) * (X.T.dot(h-y)) + (learningRate / m) * theta
    grad.flatten()
    grad0 = (1 / m) * (X.T.dot(h-y))
    grad[0,0] = grad0[0]
    return J, grad

def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X*theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    grad[0,0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    return np.array(grad).ravel() #flattens array

def pause():
    sys.stdout.flush()
    input("Hit Enter to continue")
 
def displayData(X, y, m, n, indexes, title):  
    index = 0
    for j in range(0, m):
        for i in range(0, n):
            #stack images across the row
            image = X[[indexes[index]],]
            image = image.reshape(20,20)
            image = np.flipud(image)
            image = np.rot90(image, k = 3)
            if i == 0:
                imageR = image
            else:
                imageR = np.hstack((imageR, image))
            index = index + 1
        # now append down by adding rows    
        if j == 0:
            imageS = imageR
        else:
            imageS = np.vstack((imageS, imageR))
    fig, ax = plt.subplots(1,1)
    ax.imshow(imageS)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)
    # Add label showing what the value is (or predicted)
    idx = 0
    for i in range(0, m):
        for j in range(0, n):
            value = y[indexes[idx]]
            #print(idx, indexes[idx], y[indexes[idx]])
            #value = value[0]
            if value == 10:
                value = 0
            ax.text(j*20+2, i*20+2, value,
                           ha="center", va="center", color="k",
                           fontweight="bold") 
            idx += 1

def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))
    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        # minimize the objective function
        fmin = minimize(fun=cost2, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x
    
    return all_theta

def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)
    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1
    
    return h#_argmax

print('------Exercise 3 Part 1: Loading and Visualizing Data-----')
print()
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
# Display the Data
# Exercise 3 part 1 One-vs-All
images = np.random.randint(0, 5000, size = 25)
displayData(X, y, 5, 5, images, "Sample of Dataset")   

plt.show()
pause()

print('------Exercise 3 Part 2a: One vs All with Logistic Regression \
Cost Function-----')
print()

theta = np.array([-2, -1, 1, 2])
Xt = np.array([(1.0, .1, .6, 1.1),(1.0, .2, .7, 1.2),(1.0, .3, .8, 1.3),
              (1.0, .4, .9, 1.4),(1.0, .5, 1.0, 1.5)])
yt = np.array([[1],
                [0],
                [1],
                [0],
                [1]])
yt1 = np.array([1, 0, 1, 0, 1])
lambda_t = 3

print('This is our test data set')
print(Xt)
print('This is our output')
print(yt)
cost_1, grad = costReg(theta, Xt, yt1, lambda_t)
cost_2 = cost(theta, Xt, yt1, lambda_t)
print('The cost function should be 2.534819 ')
print('The output from the LRcostfunction is ', cost_1)
print('From the modified cost function is ', cost_2)
print('The Expected gradients are:')
print('[0.14656, -0.54855, 0.724722, 1.398003]')
print('The calculated gradients are: ')
print(grad)
print()
print('-----End of Part 2a------')

pause()

print()
print('------Exercise 3 Part 2b: One-vs-All Training-----')
print('Please wait - this takes a bit')
print()

rows = data['X'].shape[0]
params = data['X'].shape[1]
all_theta = np.zeros((10, params + 1))
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
theta = np.zeros(params + 1)
lambdat = 500
all_theta = one_vs_all(data['X'], data['y'], 10, lambdat)
# one vs all uses cost2 - which aligns the vector

y_pred = predict_all(data['X'], all_theta)
# Could not get code below to work from johnwittenauer.net
# ---> correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
# ---> accuracy = (sum(map(int, correct)) / float(len(correct)))

ind = y_pred.argmax(axis = 1) + 1
count = 0
error = []
mis_label = []

for i in range(0, len(y)):
    if y[i] == ind[i]:
        count += 1                # Good - increment count
    else:  
        error.append(i)           # Record index of bad read

print('The number predicted correctly = ', count)
print('The percentage accuracy is ','{:.2%}'.format(count/len(y)))
print()
# Need clean data to plot - no 'ones'
misread = loadmat('ex3data1.mat')
misimages = random.sample(error, 25)

X = misread['X']
y = misread['y']

# Display a selection of the mis-classified
m = 0
# Display size
num_wrong = len(error)
if num_wrong >= 25:
    display_size = 25
    m = 5
    n = 5
if num_wrong < 25 and num_wrong >= 16:
    display_size = 16
    m = 4
    n = 4
if num_wrong < 16 and num_wrong >= 9:
    display_size = 9
    m = 3
    n = 3
if num_wrong < 9 and num_wrong >= 4:
    display_size = 4
    m = 2
    n = 2
if m > 0:
    misimages = random.sample(error, display_size)    
    displayData(X, ind, m, n, misimages, 'LR One-Vs-All mis-lassified')  
print('These are the mis-classified and how they were mis-classified')
print()

data = loadmat('ex3data1.mat')

plt.show()
pause()


# #-----------------------------------------------------------------------------
# # Normal Equation Method
# print('----- Normal Equation Method -----')
# print()

# lambdaN = 10
# X = data['X']
# rows = data['X'].shape[0]

# X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

# y = data['y']

# X = np.matrix(X)
# y = np.matrix(y)

# Ident = np.eye(401)
# Ident[0,0] = 0

# A = X.T*X
# B = lambdaN * Ident
# C = X.T*y

# theta = (A + B).I * C



# ind = X * theta

# found_distinct = len(np.unique(ind))

# ind = ind.round(decimals =0) * 10 / found_distinct

# count = 0
# for i in range(0, len(y)):
#     if y[i] == ind[i]:
#         count += 1                # Good - increment count
#     # else:  
#     #     error.append(i)           # Record index of bad read

# print('The number predicted correctly = ', count)
# print('The percentage accuracy is ','{:.2%}'.format(count/len(y)))
# print()

# pause()

# np.unique(ind)
# Out[4]: matrix([[ 0.,  0., -0., ..., 12., 12., 13.]])

#Sci-kit learn Multi-Class Classification Techniques
#-----------------------------------------------------------------------------
#https://scikit-learn.org/stable/modules/multiclass.html#one-vs-one

print('-----Sci-kit Learn Error-Correcting Output-Codes-----')
print()

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

y = y.T
y = y[0]

# n_classes = 10
# code_size = np.log2(n_classes) / n_classes
# yields .332

clf = OutputCodeClassifier(LinearSVC(random_state=0),
                            code_size=2, random_state=0)
ind2 = clf.fit(X, y).predict(X)
error = []
count = 0
for i in range(0, len(y)):
    if y[i] == ind2[i]:
        count += 1                # Good - increment count
    else:  
        error.append(i)           # Record index of bad read

print('The number predicted correctly = ', count)
print('The percentage accuracy is ','{:.2%}'.format(count/len(y)))
print()

# Display a selection of the mis-classified
m = 0
# Display size
num_wrong = len(error)
if num_wrong >= 25:
    display_size = 25
    m = 5
    n = 5
if num_wrong < 25 and num_wrong >= 16:
    display_size = 16
    m = 4
    n = 4
if num_wrong < 16 and num_wrong >= 9:
    display_size = 9
    m = 3
    n = 3
if num_wrong < 9 and num_wrong >= 4:
    display_size = 4
    m = 2
    n = 2
if m > 0:
    misimages = random.sample(error, display_size)    
    displayData(X, ind2, m, n, misimages, 'Sci-Kit learn Error-Correcting Output-Codes')  

plt.show()
pause()

#-----------------------------------------------------------------------------
#https://scikit-learn.org/stable/modules/multiclass.html?highlight=outputcodeclassifier
print('-----Sci-kit Learn One-Vs-The-Rest Classifier-----')
print()

clf = OneVsRestClassifier(LinearSVC(random_state=0))
#clf.verbose = True
ind3 = clf.fit(X, y).predict(X)

error =[]
count = 0
for i in range(0, len(y)):
    if y[i] == ind3[i]:
        count += 1                # Good - increment count
    else:  
        error.append(i)           # Record index of bad read

print('The number predicted correctly = ', count)
print('The percentage accuracy is ','{:.2%}'.format(count/len(y)))
print()

# Display a selection of the mis-classified
m = 0
# Display size
num_wrong = len(error)
if num_wrong >= 25:
    display_size = 25
    m = 5
    n = 5
if num_wrong < 25 and num_wrong >= 16:
    display_size = 16
    m = 4
    n = 4
if num_wrong < 16 and num_wrong >= 9:
    display_size = 9
    m = 3
    n = 3
if num_wrong < 9 and num_wrong >= 4:
    display_size = 4
    m = 2
    n = 2
if m > 0:
    misimages = random.sample(error, display_size)    
    displayData(X, ind3, m, n, misimages, 'Sci-Kit learn One-Vs-Rest misclassified')  

print('These are the mis-classified and how they were mis-classified')
print()

plt.show()
pause()

print('Saving NN portion for Exercise 4')
print('End of Exercise 3')

# for future exploration is to train the data on a 'perfect'
# data set and see what the accuracy is.



        