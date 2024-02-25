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
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

#%% simple test

k = 1000  # as much as possible

# 3 detections of levels + linear shift
w = [
      [[1 * k], [1 * k], [1 * k], [-1], [1]],
      [[80, 80, 80, -80, -80]]
      ]

biases = [
    [-0.5 * k, -1.5 * k, 0.5 * k, 0, 0],
    [0]
    ]


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


plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

print(MSE(predictions, y_test))

print(net.get_all_weights())
print(net.get_all_biases())

