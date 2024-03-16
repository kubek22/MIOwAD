#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
import copy

#%%

def save(array, file_name):
    file= open(file_name, 'wb')
    pickle.dump(array, file)
    file.close()

def read(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

#%%

df_train = read_csv("data/regression/steps-large-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/steps-large-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

#%%

plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'o', markersize=2)
plt.show()

#%% data thinning

eps = 0.05
step = 100

jump1 = (max(x_train[y_train == -80]) + min(x_train[y_train == 0])) / 2

step1 = x_train[y_train == -80]
step1 = np.concatenate((step1[step1 < jump1 - eps][::step], step1[step1 >= jump1 - eps]))
y1 = np.linspace(-80, -80, len(step1))

jump2 = (max(x_train[y_train == 0]) + min(x_train[y_train == 80])) / 2

step2 = x_train[y_train == 0]
step2 = np.concatenate((step2[(step2 > jump1 + eps) & (step2 < jump2 - eps)][::step], step2[(step2 <= jump1 + eps) | (step2 >= jump2 - eps)]))
y2 = np.linspace(0, 0, len(step2))

jump3 = (max(x_train[y_train == 80]) + min(x_train[y_train == 160])) / 2

step3 = x_train[y_train == 80]
step3 = np.concatenate((step3[(step3 > jump2 + eps) & (step3 < jump3 - eps)][::step], step3[(step3 <= jump2 + eps) | (step3 >= jump3 - eps)]))
y3 = np.linspace(80, 80, len(step3))

step4 = x_train[y_train == 160]
step4 = np.concatenate((step4[step4 > jump3 + eps][::step], step4[step4 <= jump3 + eps]))
y4 = np.linspace(160, 160, len(step4))

x_train = np.concatenate((step1, step2, step3, step4))
y_train = np.concatenate((y1, y2, y3, y4))

plt.plot(x_train, y_train, 'o')
# plt.plot(x_test, y_test, 'o', markersize=2)
plt.show()

#%%

def sigma(x):
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

def ReLU(x):
    if x > 0:
        return x
    return 0.0

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

#%% scaling

scaler_x = MinMaxScaler(feature_range=(-0.8, 0.8))
x_train_scaled = scaler_x.fit_transform(np.transpose([x_train]))
x_test_scaled = scaler_x.transform(np.transpose([x_test]))

scaler_y = MinMaxScaler(feature_range=(-0.8, 0.8))
scaler_y = MinMaxScaler(feature_range=(-1000, 1000)) # alt
y_train_scaled = scaler_y.fit_transform(np.transpose([y_train]))
y_test_scaled = scaler_y.transform(np.transpose([y_test]))

plt.plot(x_train_scaled, y_train_scaled, 'o')
plt.plot(x_test_scaled, y_test_scaled, 'o', markersize=2)
plt.show()

#%%

start = time.time()

f = [sigma, sigma, lambda x: x]
net_momentum = Net(n_neurons=[5, 5, 1], n_inputs=1, functions=f, param_init='xavier')
_w = copy.deepcopy(net_momentum.get_all_weights())
_b = copy.deepcopy(net_momentum.get_all_biases())
net_rmsprop = Net(weights=_w, biases=_b, functions=f)

epoch = 1
epochs = []
MSE_momentum = []
MSE_rmsprop = []
current_MSE = math.inf
e = 10

while current_MSE > 3:
    epochs.append(epoch)
    epoch += e
    net_momentum.fit(x_train_scaled, y_train, batch_size=1, epochs=e, alpha=0.01, # greater
                     method='momentum', m_lambda=0.9)
    net_rmsprop.fit(x_train_scaled, y_train, batch_size=1, epochs=e, alpha=0.01,
                method='rmsprop', beta=0.9)
    mse_m = count_MSE(net_momentum, x_test_scaled, y_test)
    MSE_momentum.append(mse_m)
    mse_r = count_MSE(net_rmsprop, x_test_scaled, y_test)
    MSE_rmsprop.append(mse_r)
    current_MSE = min(mse_m, mse_r)
    print("Current epoch: ", epoch - 1)
    print("MSE m: ", mse_m)
    print("MSE r: ", mse_r)
    print()
        
end = time.time()

#%%

f = [sigma, lambda x: x]
net_momentum = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

net_momentum.fit(x_train_scaled, y_train, batch_size=1, epochs=100, alpha=0.001,
                 method='momentum', m_lambda=0.9)
mse_m = count_MSE(net_momentum, x_test, y_test, scaler_y)
print("MSE m: ", mse_m)
print()

predictions = []
for x in x_test:
    predictions.append(net_momentum.predict(x))
    
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

#%%

f = [sigma, lambda x: x]
net_rmsprop = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

net_rmsprop.fit(x_train_scaled, y_train, batch_size=1, epochs=100, alpha=0.001,
                 method='momentum', m_lambda=0.9)
mse_r = count_MSE(net_rmsprop, x_test, y_test, scaler_y)
print("MSE m: ", mse_r)
print()

predictions = []
for x in x_test:
    predictions.append(net_rmsprop.predict(x))
    
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

#%% results

plt.plot(epochs, MSE_GD, 'o', markersize=3)
plt.plot(epochs, MSE_SGD, 'o', markersize=3)
plt.legend(('GD', 'SGD'), loc='upper right')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

for norm in norms:
    plt.plot(epochs, norm, 'o')
plt.legend(('layer 1', 'layer 2', 'layer 3'), loc='upper left')
plt.xlabel('epoch')
plt.ylabel('Frobenius norm')
plt.show()
    
plot_weights_on_layers(net_SGD)

predictions = []
for x in x_test_scaled:
    predictions.append(net_SGD.predict(x))
predictions = np.array(predictions)

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

print("MSE on test set: ", current_MSE_SGD)
print("Last epoch: ", epoch - 1)
print("Time elapsed: ", end - start)


