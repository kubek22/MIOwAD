#%% RESULTS
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
from numpy import array


#%%
df_train = read_csv("data/regression/square-simple-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]
y = x_train ** 2

df_test = read_csv("data/regression/square-simple-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

#%%

def sigma(x):
    return 1 / (1 + math.e ** ((-1) * x))

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)


#%% weights

w0 = array([[-2.23 ,  1.9 ,  0.08, -3.27 , -3.3   ]])
w0 = np.transpose(w0)


w1 = array([[ 0.9],
       [ 2.95],
       [ 1.43],
       [-2.68],
       [ 2.6]],)  * 90
w1 = np.transpose(w1)

b0 = array([-1.4, -2.3,  1.16,  5.67, -4])
b0 = np.transpose(b0)

b1 = array([-30])
b1 = np.transpose(b1)



w = [w0, w1]

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


plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

print(MSE(predictions, y_test))

print(net.get_all_weights())
print(net.get_all_biases())



