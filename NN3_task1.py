#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
import time
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

def sigma(x):
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

def count_MSE(net, x_test, y_test, a=1, b=0):
    predictions = []
    for x in x_test:
        predictions.append(net.predict(x))
    predictions = np.array(predictions)
    predictions = predictions * a + b
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

b = np.min(y_train)
a = np.mean((y_train - b) / (x_train ** 2))

#%% data generating

(y_train - b) / a / x_train ** 2 # quadratic dependency

train_dens = len(x_train) / (max(x_train) - min(x_train))
k = train_dens * (min(x_train) + max(x_train))
k = int(k)
x_train_add = np.linspace(-max(x_train), min(x_train), k)
y_train_add = x_train_add ** 2 * a + b

x_train = np.concatenate((x_train, x_train_add))
y_train = np.concatenate((y_train, y_train_add))

plt.plot(x_train, y_train, 'o', markersize=6)
plt.show()

#%%

start = time.time()

f = [sigma, lambda x: x]
net_momentum = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')
_w = copy.deepcopy(net_momentum.get_all_weights())
_b = copy.deepcopy(net_momentum.get_all_biases())
net_rmsprop = Net(weights=_w, biases=_b, functions=f)

epoch = 1
epochs = []
MSE_momentum = []
MSE_rmsprop = []
current_MSE = math.inf
e = 10

while current_MSE > 1:
    epochs.append(epoch)
    epoch += e
    net_momentum.fit(x_train, (y_train - b) / a, batch_size=1, epochs=e, alpha=0.003,
                     method='momentum', m_lambda=0.9)
    net_rmsprop.fit(x_train, (y_train - b) / a, batch_size=1, epochs=e, alpha=0.003,
                method='rmsprop', beta=0.9)
    mse_m = count_MSE(net_momentum, x_test, y_test, a, b)
    MSE_momentum.append(mse_m)
    mse_r = count_MSE(net_rmsprop, x_test, y_test, a, b)
    MSE_rmsprop.append(mse_r)
    current_MSE = min(mse_m, mse_r)
    print("Current epoch: ", epoch - 1)
    print("MSE m: ", mse_m)
    print("MSE r: ", mse_r)
    print()
        
end = time.time()

#%% results

plt.plot(epochs, MSE_momentum, 'o-', markersize=4)
plt.plot(epochs, MSE_rmsprop, 'o-', markersize=4)
plt.legend(('Momentum', 'RMSProp'), loc='upper left')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

plt.plot(epochs[1:], MSE_momentum[1:], 'o-', markersize=4)
plt.plot(epochs[1:], MSE_rmsprop[1:], 'o-', markersize=4)
plt.legend(('Momentum', 'RMSProp'), loc='upper left')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

predictions = []
for x in x_test:
    predictions.append(net_momentum.predict(x))
predictions = np.array(predictions) * a + b
    
plt.plot(x_test, y_test, 'o', markersize=6)
plt.plot(x_test, predictions, 'o', markersize=3)
plt.legend(('test data', 'Momentum prediction'), loc='upper left')
plt.show()

print("Current epoch: ", epoch - 1)
print("MSE m: ", mse_m)
print("MSE r: ", mse_r)

#%% weights

print(net_momentum.get_all_weights())
# [array([[-1.44975512],
#        [-2.59111504],
#        [ 1.97940641],
#        [ 0.00856362],
#        [ 2.51926735]]), array([[ 3.43419467, -4.85485462,  2.55283247,  3.87368232, -4.3562739 ]])]

print(net_momentum.get_all_biases())
# [array([-1.30008302,  4.84404562, -1.37596462,  1.91585271,  4.81962509]), array([4.50906119])]
