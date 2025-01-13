#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:32:50 2024

@author: kaveen-prabodhya
"""

# Random Forest Regression
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
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])

""" The number of steps in the scatter does not increase because of we increase the 
number of trees using, reason is the more trees can converge to the same average and it
is certainly remain to the one shape (we can expect same average multiple time and getting 
avg will stay in the same form)
when we multiply the number of trees by 10 to make the n_extimator=100 the steps in scatter 
does not increase by 10 cause the reason is above eqaul avg of avg
it can be little more but not 10 times
"""

# Visualizing the Regression results (for higher resolution and smoother cruve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regressor Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()