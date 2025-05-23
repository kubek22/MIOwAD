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

#%% data thinning

# eps = 0.001 # 0.01
# step = 100 # 200

# jump1 = (max(x_train[y_train == -80]) + min(x_train[y_train == 0])) / 2

# step1 = x_train[y_train == -80]
# step1 = np.concatenate((step1[step1 < jump1 - eps][::step], step1[step1 >= jump1 - eps]))
# y1 = np.linspace(-80, -80, len(step1))

# jump2 = (max(x_train[y_train == 0]) + min(x_train[y_train == 80])) / 2

# step2 = x_train[y_train == 0]
# step2 = np.concatenate((step2[(step2 > jump1 + eps) & (step2 < jump2 - eps)][::step], step2[(step2 <= jump1 + eps) | (step2 >= jump2 - eps)]))
# y2 = np.linspace(0, 0, len(step2))

# jump3 = (max(x_train[y_train == 80]) + min(x_train[y_train == 160])) / 2

# step3 = x_train[y_train == 80]
# step3 = np.concatenate((step3[(step3 > jump2 + eps) & (step3 < jump3 - eps)][::step], step3[(step3 <= jump2 + eps) | (step3 >= jump3 - eps)]))
# y3 = np.linspace(80, 80, len(step3))

# step4 = x_train[y_train == 160]
# step4 = np.concatenate((step4[step4 > jump3 + eps][::step], step4[step4 <= jump3 + eps]))
# y4 = np.linspace(160, 160, len(step4))

# x_train = np.concatenate((step1, step2, step3, step4))
# y_train = np.concatenate((y1, y2, y3, y4))

# plt.plot(x_train, y_train, 'o')
# # plt.plot(x_test, y_test, 'o', markersize=2)
# plt.show()

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

#%% scaling

b = np.min(y_train)
a = np.mean((y_train - b) / (x_train ** 2))

#%% basic net

f = [sigma, lambda x: x]
net = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

net.fit(x_train, (y_train - b) / a, 1, epochs=100, alpha=0.003)

predictions = []
for x in x_test:
    predictions.append(net.predict(x))
predictions = np.array(predictions)
predictions = predictions * a + b

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

#%% momentum

f = [sigma, lambda x: x]
net = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

net.fit(x_train, (y_train - b) / a, 1, epochs=100, alpha=0.003, method='momentum', m_lambda=0.9)

predictions = []
for x in x_test:
    predictions.append(net.predict(x))
predictions = np.array(predictions)
predictions = predictions * a + b

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

#%% rmsprop

f = [sigma, lambda x: x]
net = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

net.fit(x_train, (y_train - b) / a, 1, epochs=100, alpha=0.003, method='rmsprop', beta=0.9)

predictions = []
for x in x_test:
    predictions.append(net.predict(x))
predictions = np.array(predictions)
predictions = predictions * a + b

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()


