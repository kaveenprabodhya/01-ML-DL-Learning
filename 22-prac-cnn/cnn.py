#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:10:30 2024

@author: kaveen-prabodhya
"""

# CNN

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Importing the Dataset
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters=32, kernel_size=3, input_shape=(64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = 2, strides = 2))

# Adding a second convolutiona layer
classifier.add(Conv2D(filters=32, kernel_size=3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = 2, strides = 2))

# Step 3 - Flatten
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=128, activation='relu' ))

classifier.add(Dense(units=1, activation='sigmoid' ))

# Compiling CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN model to trainign set
train_ds = keras.utils.image_dataset_from_directory(
    directory='dataset/training_set/',
    labels='inferred',
    label_mode='binary',
    batch_size=32,
    image_size=(64, 64))

test_ds = keras.utils.image_dataset_from_directory(
    directory='dataset/test_set/',
    labels='inferred',
    label_mode='binary',
    batch_size=32,
    image_size=(64, 64))

classifier.fit(train_ds,
               epochs=25, 
               validation_data=test_ds, 
               validation_steps = 2000)

# Making a single prediction

from keras.preprocessing import image
test_img = image.load_img('dataset/single-prediction/cat_or_dog_01.jpg', target_size= (64, 64))
test_img = image.img_to_array(test_img)

test_img = np.expand_dims(test_img, axis=0)

result = classifier.predict(test_img)

class_names = train_ds.class_names
print(class_names)

if result[0][0] == 1:
    prediction = "Dog"
else:
    prediction = "Cat"

print(prediction)

