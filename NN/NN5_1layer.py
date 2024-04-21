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

#%%

max_epochs = 1000

#%% one layer sigmoid

f = ['sigmoid', 'linear']
net_sigma1 = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

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

save(MSE_results, 'sigma1')

#%% results

plot_results('sigma1')
plot_results('sigma1', save=True, name='plot_sigma1')

#%% one layer linear

f = ['linear', 'linear']
net_linear1 = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

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

save(MSE_results, 'linear1')

#%% results

plot_results('linear1')
plot_results('linear1', save=True, name='plot_linear1')

#%% one layer tanh

f = ['tanh', 'linear']
net_tanh1 = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

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

save(MSE_results, 'tanh1')

#%% results

plot_results('tanh1')
plot_results('tanh1', save=True, name='plot_tanh1')

#%% one layer relu

f = ['relu', 'linear']
net_relu1 = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

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

save(MSE_results, 'relu1')

#%% results

plot_results('relu1')
plot_results('relu1', save=True, name='plot_relu1')
    
#%% summary

MSE_sigmoid = read('sigma1')
MSE_linear = read('linear1')
MSE_tanh = read('tanh1')
MSE_relu = read('relu1')
epochs = np.arange(1, max_epochs + 1)

plt.plot(epochs, MSE_sigmoid)
plt.plot(epochs, MSE_linear)
plt.plot(epochs, MSE_tanh)
plt.plot(epochs, MSE_relu)
plt.legend(('sigmoid', 'linear', 'tanh','ReLU'), loc='upper right')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()  
    
print('min sigmoid MSE: ', min(MSE_sigmoid))
print('min linear MSE: ', min(MSE_linear))
print('min tanh MSE: ', min(MSE_tanh))
print('min ReLU MSE: ', min(MSE_relu))
