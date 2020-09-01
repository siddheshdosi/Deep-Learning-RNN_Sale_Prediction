# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:00:16 2020

@author: SDOSI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sale_data = pd.read_csv('lb_sale_data.csv')

sale_data.TRANSACTION_DT = pd.to_datetime(sale_data.TRANSACTION_DT)


train_data = sale_data[sale_data.TRANSACTION_DT<='2018-12-31']
training_set = train_data.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

long_back=120

X_train=[]
y_train=[]
for i in range(long_back,training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-long_back:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#regressor.add(LSTM(units=50,return_sequences=True))
#regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(X_train,y_train,epochs=300,batch_size=32)

test_data = sale_data[sale_data.TRANSACTION_DT>'2018-12-31']

total_data = pd.concat([train_data['SALES'],test_data['SALES']])
inputs = total_data[len(total_data)-len(test_data) - long_back :].values

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

for i in range(long_back,inputs.shape[0]):
    X_test.append(inputs[i-long_back:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_sale_price = regressor.predict(X_test)
predicted_sale_price = sc.inverse_transform(predicted_sale_price)

real_sale_price = test_data.iloc[:,1:2].values

plt.plot(real_sale_price,color='red',label='real sale price')
plt.plot(predicted_sale_price,color='blue',label='predicted sale price')

plt.title('LB Sale Price Prediction')
plt.xlabel('Time')
plt.ylabel('LB Sale Price')

plt.legend()
plt.show()

################################
predicted_sale_price = regressor.predict(X_test)
#predicted_sale_price = sc.transform(predicted_sale_price)

real_sale_price = sc.transform(test_data.iloc[:,1:2].values)

plt.plot(real_sale_price,color='red',label='real sale price')
plt.plot(predicted_sale_price,color='blue',label='predicted sale price')

plt.title('LB Sale Price Prediction')
plt.xlabel('Time')
plt.ylabel('LB Sale Price')

plt.legend()
plt.show()