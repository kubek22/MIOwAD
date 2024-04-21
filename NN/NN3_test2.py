#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
import time

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

df_train = read_csv("data/regression/square-large-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/square-large-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

#%%

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

plt.plot(x_train, y_train, 'o')
plt.show()

#%% scaling

b = np.min(y_train)
a = np.mean((y_train - b) / (x_train ** 2))

#%%

plt.plot(x_train, (y_train - b) / a, 'o')
plt.show()

#%% M

start = time.time()

f = [sigma, lambda x: x]
net_momentum = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')
net_momentum.fit(x_train, (y_train - b) / a, batch_size=1, epochs=10, alpha=0.003,
                 method='momentum', m_lambda=0.9)
mse_m = count_MSE(net_momentum, x_test, y_test, a, b)
        
end = time.time()

predictions = []
for x in x_test:
    predictions.append(net_momentum.predict(x))
predictions = np.array(predictions)
predictions = predictions * a + b

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

print(mse_m)
print(end - start)

#%% R

start = time.time()

f = [sigma, lambda x: x]
net_rmsprop = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

#%%
 
net_rmsprop.fit(x_train, (y_train - b) / a, batch_size=1, epochs=100, alpha=0.003,
                method='rmsprop', m_lambda=0)
count_MSE(net_rmsprop, x_train, y_train, a, b)
mse_r = count_MSE(net_rmsprop, x_test, y_test, a, b)
        
end = time.time()

predictions = []
for x in x_test:
    predictions.append(net_rmsprop.predict(x))
predictions = np.array(predictions)
predictions = predictions * a + b

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

print(mse_r)
print(end - start)

