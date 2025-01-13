#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:42:39 2024

@author: kaveen-prabodhya
"""

# K-Means Clustering

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset with pands
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find out the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.xticks(range(1, 11))
plt.show()

# Applying KMeans to the Mall Dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_kmeans ==0, 0], X[y_kmeans == 0, 1], s=100, c='red', label = 'Cluster 01')
plt.scatter(X[y_kmeans ==1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label = 'Cluster 02')
plt.scatter(X[y_kmeans ==2, 0], X[y_kmeans == 2, 1], s=100, c='green', label = 'Cluster 03')
plt.scatter(X[y_kmeans ==3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label = 'Cluster 04')
plt.scatter(X[y_kmeans ==4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label = 'Cluster 05')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending scorer 1-100')
plt.legend()
plt.show()













