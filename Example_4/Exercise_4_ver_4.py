# -*- coding: utf-8 -*-


"""
Conversion of Ex 4 from Andrew Ng's Machine Learning Course
on Coursera from Matlab to Python.  Most but not all of code is from 
johnwittenauer.net.

I created the plotting routine for matplotlib of the sample set
and mis-classified samples.

Added the NN tool from Sci-Kit learn with three different
hidden layer sizes, 10, 20, & 400.

"""


import random
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import sys

#from sklearn.multiclass import OutputCodeClassifier
#from sklearn.svm import LinearSVC
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    
    a1 = np.insert(X, 0, values = np.ones(m), axis = 1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values = np.ones(m), axis = 1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    
    return a1, z2, a2, z3, h

def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
                                        #data - slice of params
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], \
            (hidden_size, (input_size + 1))))
                # reshape size            
            
            
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], \
            (num_labels, (hidden_size + 1))))        
            
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    #compute the cost
    J=0
    for i in range(m):
        first = np.multiply(-y[i,:], np.log(h[i,:]))
        second = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first - second)
    J = J / m
    
    J += (float(learning_rate) / (2*m)) * (np.sum(np.power(theta1[:,1:], 2)) + \
            np.sum(np.power(theta2[:,1:], 2)))
    return J

def predict(theta, X):
    probability = sigmoid(X*theta.T)
    return [1 if x>= 0.5 else 0 for x in probability]
    
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

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)
        
        d3t = ht - yt  # (1, 10)
        
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
        
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return J, grad

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

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = .1

# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)
#-------------------------------
# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

# theta1.shape, theta2.shape

# Display the Data
# Exercise 3 part 1 One-vs-All
images = np.random.randint(0, 5000, size = 25)
displayData(X, y, 5, 5, images, "Sample of Dataset")   

plt.show()
pause()

# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), 
                method='TNC', jac=True, options={'maxiter': 250})

X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
ind2 = np.array(np.argmax(h, axis=1) + 1)

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
    displayData(X, ind2, m, n, misimages, 'Neural Network Mis-classifications')  

plt.show()
pause()

#-----------------------------------------------------------------------------
# Neural Network method from Scikit-learn
# solver = 'sgd'
print('Using the Sci_Kit Learn Neural Network tool with hidden layer')
print('sizes of 10, 20, & 400 and solver = "sgd" ')


def Neural_Net(X, y, HL):
    
    X1 = np.ones((5000,1))
    X = np.hstack((X1, X))
    #Multi-Layer Perceptron using Stochastic Gradient Descent
    mlp = MLPClassifier(hidden_layer_sizes=(HL,), max_iter=50, alpha=1e-4,
                        solver='sgd', verbose=False, tol=1e-4, random_state=1,
                        learning_rate_init=.5)
    # verbose = 10 shows iterations and results, stops on 10 of change less than 
    # target
    #y_flat = y.ravel()
                    
    mlp.fit(X, y)
    NN_Score = mlp.score(X, y)
    NN_Train = mlp.predict(X)
    
    return NN_Train, NN_Score


HLlist = [10, 20, 400]

for i in range(0,3):
    
    HL = HLlist[i]
    NN_Train, NN_Score = Neural_Net(X, y, HL)
    
    error =[]
    count = 0
    for i in range(0, len(y)):
        if y[i] == NN_Train[i]:
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
    if num_wrong < 4 and num_wrong >= 2:
        display_size = 2
        m = 2
        n = 1    
    if m > 0:
        print('These are the mis-classified and how they were mis-classified')
        print()
        misimages = random.sample(error, display_size)    
        Title_Str = 'Sci-Kit learn NN with hidden layer size of ' + str(HL)
        displayData(X, NN_Train, m, n, misimages, Title_Str)
        plt.show()
                      

# With Hidden Layer size of 400 the model is 100% trained to the training
# set.  Looking at the mis-classified for HL = 20 it is trained to poor
# samples.  Future work may be to train on 'perfect' samples.

print()
print('------ End of Exercise 4 ------')  
print()  
    
    
    








