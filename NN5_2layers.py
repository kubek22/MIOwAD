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

def load_net(w_src, b_src, functions):
    weights = read(w_src)
    biases = read(b_src)
    return Net(weights, functions, biases)

def plot_results(res_src, save=False, name=None):
    MSE = read(res_src)
    epochs = np.arange(1, len(MSE) + 1)
    plt.plot(epochs, MSE)
    if save:
        plt.savefig(name)
    else:
        plt.show()

#%%

df_train = read_csv("data/regression/multimodal-large-training.csv")
df_train.head()
x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/multimodal-large-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

#%% scaling

scaler_x = MinMaxScaler(feature_range=(-0.9, 0.9))
x_train_scaled = scaler_x.fit_transform(np.transpose([x_train]))
x_test_scaled = scaler_x.transform(np.transpose([x_test]))

scaler_y = MinMaxScaler(feature_range=(-0.9, 0.9))
y_train_scaled = scaler_y.fit_transform(np.transpose([y_train]))
y_test_scaled = scaler_y.transform(np.transpose([y_test]))

plt.plot(x_train_scaled, y_train_scaled, 'o')
plt.plot(x_test_scaled, y_test_scaled, 'o')
plt.show()

#%% parameters

max_epochs = 500
k = 5

#%% two layer sigmoid

f = [sigma, sigma, lambda x: x]
net_sigma1 = Net(n_neurons=[k, k, 1], n_inputs=1, functions=f, param_init='xavier')

MSE_results = []

warnings.filterwarnings('ignore') 
start_time = time.time()

for i in range(max_epochs):
    net_sigma1.fit(x_train_scaled, y_train_scaled, batch_size=1, epochs=1, alpha=0.003, method='momentum')
    mse = count_MSE(net_sigma1, x_test_scaled, y_test, scaler_y)
    MSE_results.append(mse)
    print("epoch: ", i + 1)
    print("mse: ", mse)
    print()
    
end_time = time.time()

#%% save results

save(MSE_results, 'sigma2')

#%% results

plot_results('sigma2')
plot_results('sigma2', save=True, name='plot_sigma2')

#%% two layer linear

f = [lambda x: x, lambda x: x, lambda x: x]
net_linear1 = Net(n_neurons=[k, k, 1], n_inputs=1, functions=f, param_init='xavier')

MSE_results = []

warnings.filterwarnings('ignore') 
start_time = time.time()

for i in range(max_epochs):
    net_linear1.fit(x_train_scaled, y_train_scaled, batch_size=1, epochs=1, alpha=0.003, method='momentum')
    mse = count_MSE(net_linear1, x_test_scaled, y_test, scaler_y)
    MSE_results.append(mse)
    print("epoch: ", i + 1)
    print("mse: ", mse)
    print()
    
end_time = time.time()

#%% save results

save(MSE_results, 'linear2')

#%% results

plot_results('linear2')
plot_results('linear2', save=True, name='plot_linear2')

#%% two layer tanh

f = ['tanh', 'tanh', lambda x: x]
net_tanh1 = Net(n_neurons=[k, k, 1], n_inputs=1, functions=f, param_init='xavier')

MSE_results = []

warnings.filterwarnings('ignore') 
start_time = time.time()

for i in range(max_epochs):
    net_tanh1.fit(x_train_scaled, y_train_scaled, batch_size=1, epochs=1, alpha=0.003, method='momentum')
    mse = count_MSE(net_tanh1, x_test_scaled, y_test, scaler_y)
    MSE_results.append(mse)
    print("epoch: ", i + 1)
    print("mse: ", mse)
    print()
    
end_time = time.time()

#%% save results

save(MSE_results, 'tanh2')

#%% results

plot_results('tanh2')
plot_results('tanh2', save=True, name='plot_tanh2')

#%% two layer relu

f = [ReLU, ReLU, lambda x: x]
net_relu1 = Net(n_neurons=[k, k, 1], n_inputs=1, functions=f, param_init='xavier')

MSE_results = []

warnings.filterwarnings('ignore') 
start_time = time.time()

for i in range(max_epochs):
    net_relu1.fit(x_train_scaled, y_train_scaled, batch_size=1, epochs=1, alpha=0.003, method='momentum')
    mse = count_MSE(net_relu1, x_test_scaled, y_test, scaler_y)
    MSE_results.append(mse)
    print("epoch: ", i + 1)
    print("mse: ", mse)
    print()
    
end_time = time.time()

#%% save results

save(MSE_results, 'relu2')

#%% results

plot_results('relu2')
plot_results('relu2', save=True, name='plot_relu2')
