# -*- coding: utf-8 -*-
"""
Spyder Editor

Data Preprocessing
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

column_transformer = ColumnTransformer(transformers=[
    ("onehot", OneHotEncoder(), [0]),
    ], remainder='passthrough')

X = column_transformer.fit_transform(X)

X = X.astype(float)

label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

# Spitting the data set into the training and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Here no need to do y scale becuase it in 0 and 1 no need to do feature scaling
"""sc_y = StandardScaler()
y_train = sc_y.fit_transform(Y_train.reshape(-1, 1))"""










