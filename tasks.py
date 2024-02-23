#%%
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
 
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
plt.plot(x_train, y_train, 'o')
plt.show()

#%%
def sigma(x):
    return 1 / (math.e ** ((-1) * x))

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

#%%
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'o')
plt.show()

#%%
k = 2

w = [
     [[2], [2], [2], [-2], [-2]],
     [[3 * k, 2 * k, 3 * k, 1 * k, 1.2 * k]]
     ]

biases = [-124, 0]

net1 = Net(w, [sigma, lambda x: x], biases)

net1.predict([1])
net1.predict(1)

# func = np.vectorize(net1.compute)
# np.array(x_train)
# func(np.array(x_train))

predictions = []
for x in x_train:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%% 1 layer (5)

# values = [i for i in range(-4, 0)] + [i for i in range(1, 5)]

import random

weights = None
min_MSE = math.inf
for i in range(50000):
    w = [
         [[random.random() * 10 - 5] for i in range(5)],
         [np.random.rand(5) * 5]
         ]
    biases = [-150, 0]
    net1 = Net(w, [sigma, lambda x: x], biases)
    predictions = []
    for x in x_test:
        predictions.append(net1.predict(x))
    mse = MSE(predictions, y_train)
    if mse < min_MSE:
        min_MSE = mse
        weights = net1.get_all_weights()
        
min_MSE
weights

net1 = Net(weights, [sigma, lambda x: x], biases)
predictions = []
for x in x_test:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%% 1 layer (10)

# values = [i for i in range(-4, 0)] + [i for i in range(1, 5)]

import random

weights = None
min_MSE = math.inf
for i in range(10000):
    w = [
         np.random.rand(10, 1) * 10 - 5,
         np.random.rand(1, 10) * 10 - 5
         ]
    biases = [-150, 0]
    net1 = Net(w, [sigma, lambda x: x], biases)
    predictions = []
    for x in x_test:
        predictions.append(net1.predict(x))
    mse = MSE(predictions, y_train)
    if mse < min_MSE:
        min_MSE = mse
        weights = net1.get_all_weights()
        
min_MSE
weights

net1 = Net(weights, [sigma, lambda x: x], biases)
predictions = []
for x in x_test:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%% 2 layers (5, 5)

# values = [i for i in range(-4, 0)] + [i for i in range(1, 5)]

import random

weights = None
min_MSE = math.inf
for i in range(1):
    w = [
         np.random.rand(5, 1) * 10 - 5,
         np.random.rand(5, 5) * 10 - 5,
         np.random.rand(1, 5) * 10 - 5
         ]
    biases = [0, -150, 0]
    net1 = Net(w, [sigma, sigma, lambda x: x], biases)
    predictions = []
    for x in x_test:
        predictions.append(net1.predict(x))
    mse = MSE(predictions, y_train)
    if mse < min_MSE:
        min_MSE = mse
        weights = net1.get_all_weights()
        
min_MSE
weights

net1 = Net(weights, [sigma, lambda x: x], biases)
predictions = []
for x in x_test:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%%

k = 1

w = [
     [[-2], [-2], [1.2], [1.5], [1.5]],
     [np.array([1, 2, 2, 2, 2]) * k]
     ]

w = [[[-2],
        [ 1.5],
        [ 1.7],
        [-1.7],
        [-2.3]],
 np.array([[1.1, 4.7, 3.8, 2.5, 3.7]]) * k  ]

biases = [-150, 0]

net1 = Net(w, [sigma, lambda x: x], biases)

net1.predict([1])
net1.predict(1)

# func = np.vectorize(net1.compute)
# np.array(x_train)
# func(np.array(x_train))

predictions = []
for x in x_test:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

MSE(predictions, y_train)


#%%

k = 3

w = [
     [[-2.5], [0.5], [2], [0.5], [-2.5]],
     [[1 * k, 1 * k, 2.5 * k, 0.1 * k, 1 * k]]
     ]

biases = [-130, 0]

net1 = Net(w, [sigma, lambda x: x], biases)

net1.predict([1])
net1.predict(1)

# func = np.vectorize(net1.compute)
# np.array(x_train)
# func(np.array(x_train))

predictions = []
for x in x_test:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%%
np.array(y_train)
np.sqrt(np.array(y_train))

x = np.array(x_train)
x = np.abs(x)
np.min(x)
np.where(x == np.min(x))
x[4]
y = np.array(y_train)
b = y[4]

y -= b
y

plt.plot(x, y, 'o')
plt.show()


a = np.mean((np.sqrt(y) / x) ** 2)
a
b


#%%

w = [
     [[-2.5], [0.5], [2], [0.5], [-2.5]],
     [[1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, -1], [0, 1, 0, 1, 0], [1, 0, 0, 0, -1]],
     [[1, 0, 0, 0, 0]]
     ]

biases = [0, -130, 0]

net2 = Net(w, [sigma, sigma, lambda x: x], biases)

predictions = []
for x in x_train:
    predictions.append(net2.predict(x))
    
predictions
plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)

