#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on Wed Jul 01 13:45:56 2020
Updated on Sun Oct 11 01:44:25 2020

@author: JasonKhu
"""

# (1) DATA PREPROCESSING

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.getcwd()
os.chdir("/Users/JasonKhu/Desktop/Personal Projects Folder/Consulting/ASOC Taylor Fry 2020/Part 3 - Recurrent Neural Networks")
os.getcwd()

# Importing the training set
dataset_train = pd.read_csv('train_1-7.csv')
training_set = dataset_train.iloc[:, 1:2].values # take volume (1) and time (2) columns
print(training_set) #32815 rows

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set) # normalisation - to range [0,1]

# Creating the X_train and y_train
X_train = []
y_train = []
for i in range(100, 32815): # look back 100 rows
    X_train.append(training_set_scaled[i-100:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the X_train
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# (2) BUILDING AND TRAINING THE RNN

# Importing the Keras libraries and packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Second LSTM layer + Dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Third LSTM layer + Dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Fourth LSTM layer + Dropout
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units=1)) # Dense used for output layer

regressor.summary()

# Compiling the RNN
regressor.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)

# (3) MAKING THE PREDICTIONS

# Importing the testing set
dataset_test = pd.read_csv('test_1-7.csv')
real_demand = dataset_test.iloc[:, 1:2].values

# Making predictions using the testing set
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test):]
inputs = inputs.iloc[:,1:2].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(100, 10176):
    X_test.append(inputs[i-100:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test
predicted_demand = regressor.predict(X_test)
predicted_demand = sc.inverse_transform(predicted_demand)
print(predicted_demand)


# (4) VISUALISING THE RESULTS

f, axs = plt.subplots(2,1,figsize=(20,5))
plt.subplot(1,2, 1)
plt.plot(real_demand, color = 'red', label = 'Real Electricity Demand')
plt.title('Real Electricity Demand')
plt.xlabel('Time')
plt.ylabel('Electricity Demand')
plt.subplot(1,2, 2)
plt.plot(predicted_demand, color = 'blue', label = 'Predicted Electricity Demand')
plt.title('Electricity Demand Prediction')
plt.xlabel('Time')
plt.ylabel('Electricity Demand')
plt.show()

# Examine the predictive MSE

import math
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(real_demand[100:len(real_demand)], predicted_demand)
mse #149326.147...
rpmse = math.sqrt(mse)
rpmse #386.427...



