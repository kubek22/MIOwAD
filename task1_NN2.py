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

df_train = read_csv("data/regression/square-simple-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/square-simple-test.csv")
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

#%%

plt.plot(x_train, y_train, 'o')
plt.show()

#%% scaling

b = np.min(y_train)
a = np.mean((y_train - b) / (x_train ** 2))

#%%

plt.plot(x_train, (y_train - b) / a, 'o')
plt.show()


#%%

f = [sigma, lambda x: x]

net = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

#%%

# w = read("weights.txt")
# b = read("biases.txt")

# net = Net(weights=w, biases=b, functions=f)

#%%

start = time.time()
net.fit(x_train, (y_train - b) / a, batch_size=16, epochs=10000, alpha=0.003) # lower batch size
end = time.time()
print("Time elapsed: ", end - start)

#%%

predictions = []
for x in x_test:
    predictions.append(net.predict(x))
    
predictions = np.array(predictions)

predictions = predictions * a + b

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

print(MSE(predictions, y_test))

#%% wartosci norm wag na warstwach

def plot_weights(net, with_bias=True):
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
    plt.show()

#%%

plot_weights(net)

#%%

def count_MSE(net, x_test, y_test, a, b):
    predictions = []
    for x in x_test:
        predictions.append(net.predict(x))
    predictions = np.array(predictions)
    predictions = predictions * a + b
    return MSE(predictions, y_test)

#%% convergence comparison

f = [sigma, lambda x: x]
net_GD = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')
net_SGD = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

n = 1000
epochs = np.arange(n)
MSE_GD = []
MSE_SGD = []
for epoch in epochs:
    net_GD.fit(x_train, (y_train - b) / a, batch_size=len(x_train), epochs=1, alpha=0.003)
    net_SGD.fit(x_train, (y_train - b) / a, batch_size=1, epochs=1, alpha=0.003)
    MSE_GD.append(count_MSE(net_GD, x_test, y_test, a, b))
    MSE_SGD.append(count_MSE(net_SGD, x_test, y_test, a, b))

plt.plot(epochs, MSE_GD, 'o')
plt.plot(epochs, MSE_SGD, 'o')
plt.legend(('GD', 'SGD'), loc='upper right')
plt.show()
    

#%%

save(net.get_all_weights(), "weights.txt")
save(net.get_all_biases(), "biases.txt")


#%%

f = [sigma, lambda x: x]

net = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

net.fit(x_train, y_train, batch_size=10, epochs=100, alpha=1)
net.fit(x_train, y_train, batch_size=10, epochs=100, alpha=0.5)
net.fit(x_train, y_train, batch_size=10, epochs=100, alpha=0.1)
net.fit(x_train, y_train, batch_size=10, epochs=100, alpha=0.05)
net.fit(x_train, y_train, batch_size=10, epochs=1000, alpha=0.01)
net.fit(x_train, y_train, batch_size=10, epochs=1000, alpha=0.005)

predictions = []
for x in x_train:
    predictions.append(net.predict(x))
    
predictions

plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

print(MSE(predictions, y_train))

net.get_all_weights()

#%%
import pandas as pd

X = pd.DataFrame([[1, 2], [3, 4]])
X
X.to_list()

X = pd.Series([[1, 2], [3, 4]])
X.to_list()

#%%

df_train[0:10]
df_train[0:10]

for x in df_train[0:10]:
    print(x)
    
np.array(np.array(df_train)[0:10])
y_train
np.array(y_train)

#%%

np.array(df_train)

for x in np.array(df_train)[0:2]:
    print(x[0])
    print(x[1])
    print(x[2])
    print(x)
    print(type(x))
    
    
