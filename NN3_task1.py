#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
import time
from sklearn.preprocessing import MinMaxScaler

#%%

def save(array, file_name):
    file= open(file_name, 'wb')
    pickle.dump(array, file)
    file.close()

def read(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

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

#%%

df_train = read_csv("data/regression/square-large-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/square-large-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

#%%

plt.plot(x_train, y_train, 'o', markersize=6)
plt.plot(x_test, y_test, 'o', markersize=4)
plt.show()

#%% scaling

scaler_x = MinMaxScaler(feature_range=(-0.8, 0.8))
x_train_scaled = scaler_x.fit_transform(np.transpose([x_train]))
x_test_scaled = scaler_x.transform(np.transpose([x_test]))

scaler_y = MinMaxScaler(feature_range=(-0.8, 0.8))
y_train_scaled = scaler_y.fit_transform(np.transpose([y_train]))
y_test_scaled = scaler_y.transform(np.transpose([y_test]))

plt.plot(x_train_scaled, y_train_scaled, 'o')
plt.plot(x_test_scaled, y_test_scaled, 'o')
plt.show()

#%%

start = time.time()

f = [sigma, lambda x: x]
net_momentum = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')
net_rmsprop = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

epoch = 1
epochs = []
MSE_momentum = []
MSE_rmsprop = []
current_MSE = math.inf
e = 10
min_learning_rate = 0.0001
lr = min_learning_rate

while current_MSE > 1:
    epochs.append(epoch)
    epoch += e
    net_momentum.fit(x_train_scaled, y_train_scaled, batch_size=16, epochs=e, alpha=lr,
                     method='momentum', m_lambda=0.9)
    net_rmsprop.fit(x_train_scaled, y_train_scaled, batch_size=16, epochs=e, alpha=0.0001,
                method='rmsprop', beta=0.9)
    mse_m = count_MSE(net_momentum, x_test_scaled, y_test, scaler_y)
    if mse_m < 100:
        lr = min_learning_rate * math.sqrt(mse_m / epoch)
    else:
        lr = 0.01
    MSE_momentum.append(mse_m)
    mse_r = count_MSE(net_rmsprop, x_test_scaled, y_test, scaler_y)
    MSE_rmsprop.append(mse_r)
    current_MSE = min(mse_m, mse_r)
    print("Current epoch: ", epoch - 1)
    print("MSE m: ", mse_m)
    print("MSE r: ", mse_r)
    print()
        
end = time.time()

#%% 

f = [sigma, lambda x: x]
net_rmsprop = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')
net_rmsprop.fit(x_train_scaled, y_train_scaled, batch_size=16, epochs=1000, alpha=0.003,
            method='rmsprop', beta=0.9)
mse_r = count_MSE(net_rmsprop, x_test_scaled, y_test, scaler_y)
print("MSE r: ", mse_r)
print()

predictions = []
for x in x_test_scaled:
    predictions.append(net_rmsprop.predict(x))
predictions_scaled = scaler_y.inverse_transform(np.array([predictions]))[0]

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions_scaled, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

plt.plot(x_test_scaled, y_test_scaled, 'o')
plt.plot(x_test_scaled, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

#%%

predictions = []
for x in x_train_scaled:
    predictions.append(net_rmsprop.predict(x))
predictions_scaled = scaler_y.inverse_transform(np.array([predictions]))[0]

plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions_scaled, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

plt.plot(x_train_scaled, y_train_scaled, 'o')
plt.plot(x_train_scaled, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()
