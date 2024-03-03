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

def sigma(x):
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

#%%

plt.plot(x_train, y_train, 'o')
plt.show()

#%%

f = [sigma, lambda x: x]

net = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

net.fit(x_train, y_train, batch_size=10, epochs=10000, alpha=0.003)

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
    
    
