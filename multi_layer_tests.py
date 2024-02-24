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
    return 1 / (1 + math.e ** ((-1) * x))

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

#%%
w = [
     [[-2.5], [0.5], [2], [0.5], [-2.5]],
     [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
     [[1, 0]]
     ]

#  biases = [0, -130, 0]

net2 = Net(w, [sigma, sigma, lambda x: x])

predictions = []
for x in x_train:
    predictions.append(net2.predict(x))
    
predictions
plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%%




np.array([4.288566453033124670e-01,
-8.577132906066249618e-02,
-3.430853162426499847e-01,
-8.577132906066249618e-02,
4.288566453033124670e-01])


f = np.vectorize(sigma)
f(0)

sigma(0)

f([1, 2, 3])
f(np.array([1, 2, 3]))






