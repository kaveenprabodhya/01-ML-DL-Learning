#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:22:26 2024

@author: kaveen-prabodhya
"""

# Hierarchical Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Cusotmers')
plt.ylabel('Euclidean Distance')
plt.show()

# Fitting Hierarchycal clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5)
y_hc = hc.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_hc ==0, 0], X[y_hc == 0, 1], s=100, c='red', label = 'Cluster 01')
plt.scatter(X[y_hc ==1, 0], X[y_hc == 1, 1], s=100, c='blue', label = 'Cluster 02')
plt.scatter(X[y_hc ==2, 0], X[y_hc == 2, 1], s=100, c='green', label = 'Cluster 03')
plt.scatter(X[y_hc ==3, 0], X[y_hc == 3, 1], s=100, c='cyan', label = 'Cluster 04')
plt.scatter(X[y_hc ==4, 0], X[y_hc == 4, 1], s=100, c='magenta', label = 'Cluster 05')
plt.title('Cluster of clients')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending scorer 1-100')
plt.legend()
plt.show()



