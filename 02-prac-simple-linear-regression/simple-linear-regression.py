#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 17:13:12 2024

@author: kaveen-prabodhya
"""

# Simple Linear Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Spitting the data set into the training and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# Based on the experience we gonna find out the salary
regressor.fit(X_train, Y_train)

# Predicting the test results
y_pred = regressor.predict(X_test)

# Visualizing the Training set Results
plt.scatter(X_train, Y_train, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set Results
plt.scatter(X_test, Y_test, color= 'red')
# =============================================================================
# We are getting the same line in plot code becasue regressor is already trained,
# and got it's equation and used that same eqation in both cases below plot creating.
# =============================================================================
# plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

