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

df_train = read_csv("data/regression/steps-small-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/steps-small-test.csv")
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

#%%

start = time.time()
net.fit(x_train, y_train, batch_size=16, epochs=1000, alpha=0.003)
end = time.time()
print("Time elapsed: ", end - start)

#%%

predictions = []
for x in x_test:
    predictions.append(net.predict(x))
    
predictions = np.array(predictions)

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

print(MSE(predictions, y_test))
