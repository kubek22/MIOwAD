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

def save(array, file_name):
    file = open(file_name, 'wb')
    pickle.dump(array, file)
    file.close()

def read(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

#%%

df_train = read_csv("data/regression/square-simple-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/square-simple-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

#%%

def ReLU(x):
    if x > 0:
        return x
    return 0.0

def sigma(x):
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

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

def plot_weights_on_layers(net, with_bias=True):
    layers = []
    norms = []
    i = 0
    for weights, biases in zip(net.get_all_weights(), net.get_all_biases()):
        layers.append(i)
        i += 1
        if with_bias:
            norms.append(np.linalg.norm(np.c_[weights, biases]))
        else:
            norms.append(np.linalg.norm(weights))
    plt.plot(layers, norms, 'o')
    plt.xlabel('layer')
    plt.ylabel('Frobenius norm')
    plt.show()


#%%

plt.plot(x_train, y_train, 'o')
plt.show()

#%% basic net

f = [sigma, lambda x: x]
net = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

net.fit(x_train, y_train, 4, epochs=100, alpha=0.003)

predictions = []
for x in x_test:
    predictions.append(net.predict(x))
predictions = np.array(predictions)
predictions = predictions

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

#%% momentum

f = [sigma, lambda x: x]
net = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

net.fit(x_train, y_train, 4, epochs=100, alpha=0.003, method='momentum', m_lambda=0.5)

predictions = []
for x in x_test:
    predictions.append(net.predict(x))
predictions = np.array(predictions)
predictions = predictions

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()


#%% rmsprop

f = [sigma, lambda x: x]
net = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

net.fit(x_train, y_train, 4, epochs=100, alpha=0.003, method='rmsprop', beta=0.5)

predictions = []
for x in x_test:
    predictions.append(net.predict(x))
predictions = np.array(predictions)
predictions = predictions

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()


