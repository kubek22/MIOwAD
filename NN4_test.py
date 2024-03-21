#%% 

import numpy as np
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
import warnings 

#%%

def save(array, file_name):
    file = open(file_name, 'wb')
    pickle.dump(array, file)
    file.close()

def read(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

def ReLU(x):
    if x > 0:
        return x
    return 0.0

def sigma(x):
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

def predict_class(predictions):
    classes = []
    for p in predictions:
        classes.append(np.where(np.max(p) == p))
    return np.array(classes).reshape(-1)

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

def count_MSE(net, x_test, y_test, scaler_y=None):
    predictions = predict(net, x_test)
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(np.array([predictions]))[0]
        res = np.array([])
    # for p in predictions:
    #     res.add()
    return MSE(predictions, y_test)

def predict(net, x_data):
    predictions = []
    for x in x_data:
        predictions.append(net.predict(x))
    return np.array(predictions)

#%%

df_train = read_csv("data/classification/easy-training.csv")
df_train.head()

xy_train = df_train[["x", "y"]].to_numpy()
c_train = df_train["c"]
c_train.loc[c_train == True] = 1
c_train.loc[c_train == False] = 0
c_train = c_train.to_numpy()

df_test = read_csv("data/classification/easy-test.csv")
df_test.head()

xy_test = df_test[["x", "y"]].to_numpy()
c_test = df_test["c"]
c_test.loc[c_test == True] = 1
c_test.loc[c_test == False] = 0
c_test = c_test.to_numpy()

#%%
        
plt.scatter(xy_train[:,0], xy_train[:,1], c=(c_train+0.5)/2)
plt.show()

#%%

f = [sigma, "softmax"]
net = Net(n_neurons=[3, 2], n_inputs=2, functions=f, param_init='xavier', use_softmax=True)
epoch = 1

#%%

epoch += 1
net.fit(xy_train, c_train, batch_size=1, epochs=1, alpha=0.003,
                  method='momentum', m_lambda=0.9)
# mse = count_MSE(net, xy_test, c_test)
# print("Current epoch: ", epoch - 1)
# print("MSE: ", mse)
# print()

#%%

preds = predict(net, xy_train)
classes = predict_class(preds)

plt.scatter(xy_train[:,0], xy_train[:,1], c=(classes+0.5)/2)
plt.show()
