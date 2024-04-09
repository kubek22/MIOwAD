#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
import warnings

#%%

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

def count_MSE(net, x_test, y_test, scaler_y=None):
    predictions = []
    for x in x_test:
        predictions.append(net.predict(x))
    predictions = np.array(predictions)
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(np.array([predictions]))[0]
    return MSE(predictions, y_test)
        
#%%

df_train = read_csv("data/regression/steps-large-training.csv")
df_train.head()
x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/steps-large-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

#%% scaling

scaler_x = MinMaxScaler(feature_range=(-0.8, 0.8))
x_train_scaled = scaler_x.fit_transform(np.transpose([x_train]))
x_test_scaled = scaler_x.transform(np.transpose([x_test]))

#%% L1

f = ['tanh', 'tanh', 'linear']
net = Net(n_neurons=[5, 5, 1], n_inputs=1, functions=f, param_init='xavier')

for i in range(100):
    net.fit(x_train_scaled, y_train, batch_size=1, epochs=1, alpha=0.001, method='rmsprop', regularization='l1', reg_lambda=0.1)
    mse = count_MSE(net, x_test_scaled, y_test)
    print("mse : ", mse)
    print()

#%% L2

f = ['tanh', 'tanh', 'linear']
net = Net(n_neurons=[5, 5, 1], n_inputs=1, functions=f, param_init='xavier')

for i in range(100):
    net.fit(x_train_scaled, y_train, batch_size=1, epochs=1, alpha=0.001, method='rmsprop', regularization='l2', reg_lambda=0.1)
    mse = count_MSE(net, x_test_scaled, y_test)
    print("mse : ", mse)
    print()
    
#%%
    
f = ['tanh', 'tanh', 'linear']
net = Net(n_neurons=[5, 5, 1], n_inputs=1, functions=f, param_init='xavier')

net.fit_until_rise(2, x_train, y_train, x_test, y_test, None, 1, 100, 0.003,
                   method='rmsprop', m_lambda=0.5, beta=0.9, 
                   regularization='l1', reg_lambda=0.2,
                   print_results=True)

