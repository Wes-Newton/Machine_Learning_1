# -*- coding: utf-8 -*-
"""
This is example 7 from Andrew Ng's
Machine Learning Course
The frist part is the python implementation
found at www.johnwittenauer.net


"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

from sklearn.cluster import KMeans

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    
    return idx

def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids

def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    
    return idx, centroids

def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    
    for i in range(k):
        centroids[i,:] = X[idx[i],:]
    
    return centroids

def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()
    
    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    
    # perform SVD
    U, S, V = np.linalg.svd(cov)
    
    return U, S, V

def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)

def pause():
    sys.stdout.flush()
    input("Hit Enter to continue")

def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

data = loadmat('ex7data2.mat')
X = data['X']

# initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
# i assume this is what jwn wanted to do 
initial_centroids = init_centroids(X, 3) 

# idx = find_closest_centroids(X, initial_centroids)
# compute_centroids(X, idx, 3)

idx, centroids = run_k_means(X, initial_centroids, 10)

cluster1 = X[np.where(idx == 0)[0],:]
cluster2 = X[np.where(idx == 1)[0],:]
cluster3 = X[np.where(idx == 2)[0],:]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
title = 'ML Exercise 6 JWN K means - Number of Clusters = 3'
plt.title(title)

ax.legend()

plt.show()

pause()

image_data = loadmat('bird_small.mat')
# print(image_data)

A = image_data['A']

# A.shape
# normalize value ranges
A = A / 255.
image = plt.imshow(A)
plt.plot()

pause()

# reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))

# randomly initialize the centroids
initial_centroids = init_centroids(X, 16)

# run the algorithm
idx, centroids = run_k_means(X, initial_centroids, 10)

# get the closest centroids one last time
idx = find_closest_centroids(X, centroids)

# map each pixel to the centroid value
X_recovered = centroids[idx.astype(int),:]

# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

# plt.imshow(X_recovered)
image.set_data(X_recovered)
# only shows last - comment out to see original image

pause()

data = loadmat('ex7data1.mat')
X = data['X']

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()

pause()

U, S, V = pca(X)

Z = project_data(X, U, 1)
print(Z)

X_recovered = recover_data(Z, U, 1)
print(X_recovered)

fig, ax = plt.subplots(figsize=(12,8))
x = np.array(X_recovered[:,0])
y = np.array(X_recovered[:,1])
ax.scatter(x,y)

plt.show()
pause()

faces = loadmat('ex7faces.mat')
X = faces['X']
#X.shape

U, S, V = pca(X)
print(U, S, V)

Z = project_data(X, U, 100)

X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[3,:], (32, 32))
plt.imshow(face)

pause()

# ---- end of jwn code ----
# skipped the 'faces' portion
# try the first part with Scikit-learn tools
# this is a useful routine!

data = loadmat('ex7data2.mat')
X = data['X']
print('Review the raw data')
print()
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1], s=30, color='k', label='Cluster 1')
ax.legend()

plt.show()

print('Examine the data and enter the estimated number of clusters ')
n_clusters = input(': ')
print()
# careful - no error trapping
n_clusters = int(n_clusters)

kmeans = KMeans(n_clusters = n_clusters).fit(X)
idx = kmeans.predict(X)

fig, ax = plt.subplots(figsize=(12,8))
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'bg', 'o']
for i in range(n_clusters):
    cluster = X[np.where(idx == i)[0],:]
    label = 'Cluster ' + str(i+1)
    ax.scatter(cluster[:,0], cluster[:,1], s=30, color=colors[i], label=label)
ax.legend()
title = 'Scikit-Learn K means - Number of Clusters = ' + str(n_clusters)
plt.title(title)
plt.show()
print('End of Exercise 7')
print()

# See the file 'Python_ML_Ch_5_part_1_PCA_ver_X.py' for
# a better PCA example





