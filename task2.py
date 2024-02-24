#%%
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
from numpy import array


#%% steps-large

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
plt.show()

#%%

plt.plot(x_test, y_test, 'o')
plt.show()

#%%
def sigma(x):
    return 1 / (1 + math.e ** ((-1) * x))

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

#%%

w0 = np.array([[1], [1], [1], [1], [1]]) * 2
w1 = np.array([[-80, 50, 50, 50, 50]]) * 1.5

w = [w0, w1]

b0 = np.array([10, 0.5, -0.5, -1.5, 0])
b1 = 0

biases = [b0, b1]

net = Net(w, [sigma, lambda x: x], biases)

predictions = []
for x in x_train:
    predictions.append(net.predict(x))
    
predictions
predictions = np.array(predictions)

plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

print(MSE(predictions, y_train))

# test

predictions = []
for x in x_test:
    predictions.append(net.predict(x))
    
predictions


# plt.plot(x_test, y_test, 'o')
# plt.plot(x_test, predictions, 'o')
# plt.show()

print(MSE(predictions, y_test))

# print(net.get_all_weights())
# print(net.get_all_biases())
