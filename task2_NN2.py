#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
import time
from sklearn.preprocessing import MinMaxScaler

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

x_train = np.array(df_train["x"])
y_train = np.array(df_train["y"])

df_test = read_csv("data/regression/steps-small-test.csv")
df_test.head()

x_test = np.array(df_test["x"])
y_test = np.array(df_test["y"])

#%%

def sigma(x):
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

def count_MSE(net, x_test, y_test, scaler_y=None):
    predictions = []
    for x in x_test:
        predictions.append(net.predict(x))
    predictions = np.array(predictions)
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(np.array([predictions]))[0]
    return MSE(predictions, y_test)

#%%

plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'o')
plt.show()

#%% generating additional values

k = 10
eps = 1/100
d = 0.05

np.unique(y_train)

x_add = np.array([])
y_add = np.array([])
break1 = (np.max(x_train[y_train == -80]) + np.min(x_train[y_train == 0])) / 2 - d
x_add = np.concatenate((x_add, np.linspace(np.max(x_train[y_train == -80]), break1 - eps, k)))
x_add = np.concatenate((x_add, np.linspace(break1 + eps, np.min(x_train[y_train == 0]), k)))
y_add = np.concatenate((y_add, np.linspace(-80, -80, k)))
y_add = np.concatenate((y_add, np.linspace(0, 0, k)))

break2 = (np.max(x_train[y_train == 0]) + np.min(x_train[y_train == 80])) / 2 - d
x_add = np.concatenate((x_add, np.linspace(np.max(x_train[y_train == 0]), break2 - eps, k)))
x_add = np.concatenate((x_add, np.linspace(break2 + eps, np.min(x_train[y_train == 80]), k)))
y_add = np.concatenate((y_add, np.linspace(0, 0, k)))
y_add = np.concatenate((y_add, np.linspace(80, 80, k)))

break3 = (np.max(x_train[y_train == 80]) + np.min(x_train[y_train == 160])) / 2 - d
x_add = np.concatenate((x_add, np.linspace(np.max(x_train[y_train == 80]), break3 - eps, k)))
x_add = np.concatenate((x_add, np.linspace(break3 + eps, np.min(x_train[y_train == 160]), k)))
y_add = np.concatenate((y_add, np.linspace(80, 80, k)))
y_add = np.concatenate((y_add, np.linspace(160, 160, k)))

x_train = np.concatenate((x_train, x_add))
y_train = np.concatenate((y_train, y_add))

#%% scaling

scaler_x = MinMaxScaler(feature_range=(-0.8, 0.8))
x_train_scaled = scaler_x.fit_transform(np.transpose([x_train]))
x_test_scaled = scaler_x.transform(np.transpose([x_test]))

plt.plot(x_train_scaled, y_train, 'o')
plt.plot(x_test_scaled, y_test, 'o')
plt.show()

#%%

f = [sigma, sigma, lambda x: x]
net = Net(n_neurons=[5, 5, 1], n_inputs=1, functions=f, param_init='xavier')
epoch = 0

#%%

start = time.time()

current_MSE = math.inf
while current_MSE > 4:
    net.fit(x_train_scaled, y_train, batch_size=1, epochs=100, alpha=0.001)
    epoch += 100
    current_MSE = count_MSE(net, x_test_scaled, y_test)
    print()
    print(current_MSE)
    print(epoch)
    
end = time.time()

#%%

predictions = []
for x in x_test_scaled:
    predictions.append(net.predict(x))
    
predictions = np.array(predictions)

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'prediction'), loc='upper left')
plt.show()

print("MSE on test set: ", current_MSE)
print("Last epoch: ", epoch)
print("Time elapsed: ", end - start)

#%%

predictions = []
for x in x_train_scaled:
    predictions.append(net.predict(x))
    
predictions = np.array(predictions)

plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.legend(('train data', 'prediction'), loc='upper left')
plt.show()

#%%

print(net.get_all_weights())

# [array([[  61.93576474],
#         [ 130.07547676],
#         [  91.54616894],
#         [ 162.08861522],
#         [-133.07079102]]), array([[ -0.68358782,   4.23320536,   5.18156445,  28.92826691,
#           -5.30175373],
#         [ 11.76158695,  21.9798658 , -10.80115653,   4.05282524,
#         -14.4607159 ],
#         [ -4.94746115,  15.11984581,  13.95792541,  -3.43070877,
#           -4.76090865],
#         [  2.10624244,   2.94437684,   3.12186382,  17.7065442 ,
#         -18.9827919 ],
#         [  2.14255029,   2.69070948,   3.0881394 ,   4.05327652,
#         -27.43692015]]), array([[67.32086457, 47.02607187, 46.95173686, 37.61049278, 42.05187561]])]
                                  
print(net.get_all_biases())

# [array([-33.08260402, -71.42652726, -41.87731959, -10.59126221,
#         -56.05604632]), array([-16.75779692,  -4.90152066, -10.51791342,   8.12430246,
#         11.67954216]), array([-80.17236565])]

#%%

save(net.get_all_weights(), "weights2.txt")
save(net.get_all_biases(), "biases2.txt")