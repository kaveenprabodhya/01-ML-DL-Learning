#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:00:20 2024

@author: kaveen-prabodhya
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import theano
import tensorflow as tf

# Importing the Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

label_encoder_X_1 = LabelEncoder()
label_encoder_X_2 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])

column_transformer = ColumnTransformer(transformers=[
    ("onehot", OneHotEncoder(), [1]),
    ], remainder='passthrough')

X = column_transformer.fit_transform(X)

X = X.astype(float)

X = X[:, 1:]

# Spitting the data set into the training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Classification to the Training Set

# Importing Keras Libraries
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(tf.keras.Input(shape=(11, )))

classifier.add(Dense(units=6, activation='relu' , kernel_initializer='uniform'))

# adding the second hidden layer
classifier.add(Dense(units=6, activation='relu' , kernel_initializer='uniform'))

# Adding the output layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN model to trainign set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Making the predictions and evaluating the model

# Predicitng the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
