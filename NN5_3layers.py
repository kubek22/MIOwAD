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

max_epochs = 300
k = 5

#%% three layer sigmoid

f = [sigma, sigma, sigma, lambda x: x]
net_sigma1 = Net(n_neurons=[k, k, k, 1], n_inputs=1, functions=f, param_init='xavier')

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

save(MSE_results, 'sigma3')

#%% results

plot_results('sigma3')
plot_results('sigma3', save=True, name='plot_sigma3')

#%% three layer linear

f = [lambda x: x, lambda x: x, lambda x: x, lambda x: x]
net_linear1 = Net(n_neurons=[k, k, k, 1], n_inputs=1, functions=f, param_init='xavier')

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

save(MSE_results, 'linear3')

#%% results

plot_results('linear3')
plot_results('linear3', save=True, name='plot_linear3')

#%% three layer tanh

f = ['tanh', 'tanh', 'tanh', lambda x: x]
net_tanh1 = Net(n_neurons=[k, k, k, 1], n_inputs=1, functions=f, param_init='xavier')

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

save(MSE_results, 'tanh3')

#%% results

plot_results('tanh3')
plot_results('tanh3', save=True, name='plot_tanh3')

#%% three layer relu

f = [ReLU, ReLU, ReLU, lambda x: x]
net_relu1 = Net(n_neurons=[k, k, k, 1], n_inputs=1, functions=f, param_init='xavier')

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

save(MSE_results, 'relu3')

#%% results

plot_results('relu3')
plot_results('relu3', save=True, name='plot_relu3')

#%% summary

MSE_sigmoid = read('sigma3')
MSE_linear = read('linear3')
MSE_tanh = read('tanh3')
MSE_relu = read('relu3')
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

#%% all results

MSE_sigmoid1 = read('sigma1')[:300]
MSE_tanh1 = read('tanh1')[:300]
MSE_relu1 = read('relu1')[:300]

MSE_sigmoid2 = read('sigma2')[:300]
MSE_tanh2 = read('tanh2')[:300]
MSE_relu2 = read('relu2')[:300]

MSE_sigmoid3 = read('sigma3')
MSE_tanh3 = read('tanh3')
MSE_relu3 = read('relu3')

epochs = np.arange(1, 300 + 1)

plt.plot(epochs, MSE_sigmoid1)
plt.plot(epochs, MSE_tanh1)
plt.plot(epochs, MSE_relu1)

plt.plot(epochs, MSE_sigmoid2)
plt.plot(epochs, MSE_tanh2)
plt.plot(epochs, MSE_relu2)

plt.plot(epochs, MSE_sigmoid3)
plt.plot(epochs, MSE_tanh3)
plt.plot(epochs, MSE_relu3)

plt.legend(('sigmoid1', 'tanh1','ReLU1',
            'sigmoid2', 'tanh2','ReLU2',
            'sigmoid3', 'tanh3','ReLU3'), loc='upper right')
plt.xlabel('epoch')
plt.ylabel('MSE')

plt.show()  

