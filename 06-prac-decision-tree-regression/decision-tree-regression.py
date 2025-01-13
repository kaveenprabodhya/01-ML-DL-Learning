#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:52:28 2024

@author: kaveen-prabodhya
"""

# Decision Tree Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Spitting the data set into the training and Test set
"""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(Y_train)"""


# Fitting the Regression Model to the Dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
# X is matrice of feature and Y is dependent varaible vector
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])

# This is non linear and non continous model, so we  can not visual as this in the below.
# Here we are plotting the resolution for 10 points which is incremnt by 1
# and here it's only plotting the predictions of the 10 salaries correspoding to the 10 levels
# and this joins the predictions with a straight line because it had no predictions to plot
# for the intervel values (interval means space between two dots) of the independent level (X).
# Visualizing the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Regression results (for higher resolution and smoother cruve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




