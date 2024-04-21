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

def plot_weights_on_layers(net, with_bias=True):
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
    plt.xlabel('layer')
    plt.ylabel('Frobenius norm')
    plt.show()

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

#%% GD and SGD comparison

start = time.time()

f = [sigma, sigma, lambda x: x]
net_GD = Net(n_neurons=[5, 5, 1], n_inputs=1, functions=f, param_init='xavier')
net_SGD = Net(n_neurons=[5, 5, 1], n_inputs=1, functions=f, param_init='xavier')

epoch = 1
epochs = []
MSE_GD = []
MSE_SGD = []
current_MSE_SGD = math.inf
norms = [[] for _ in range(net_SGD.get_n_layers())]

while current_MSE_SGD > 4:
    epochs.append(epoch)
    epoch += 1
    net_GD.fit(x_train_scaled, y_train, batch_size=len(x_train), epochs=1, alpha=0.001)
    net_SGD.fit(x_train_scaled, y_train, batch_size=1, epochs=1, alpha=0.001)
    MSE_GD.append(count_MSE(net_GD, x_test_scaled, y_test))
    current_MSE_SGD = count_MSE(net_SGD, x_test_scaled, y_test)
    MSE_SGD.append(current_MSE_SGD)
    for norm, weights, biases in zip(norms, net_SGD.get_all_weights(), net_SGD.get_all_biases()):
        norm.append(np.linalg.norm(np.c_[weights, biases]))
        
end = time.time()

#%% results

plt.plot(epochs, MSE_GD, 'o', markersize=3)
plt.plot(epochs, MSE_SGD, 'o', markersize=3)
plt.legend(('GD', 'SGD'), loc='upper right')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

for norm in norms:
    plt.plot(epochs, norm, 'o')
plt.legend(('layer 1', 'layer 2', 'layer 3'), loc='upper left')
plt.xlabel('epoch')
plt.ylabel('Frobenius norm')
plt.show()
    
plot_weights_on_layers(net_SGD)

predictions = []
for x in x_test_scaled:
    predictions.append(net_SGD.predict(x))
predictions = np.array(predictions)

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

print("MSE on test set: ", current_MSE_SGD)
print("Last epoch: ", epoch - 1)
print("Time elapsed: ", end - start)

#%%

print(net_SGD.get_all_weights())

# [array([[ 76.96532544],
#        [ 61.50798303],
#        [-80.74217717],
#        [ 55.70524017],
#        [-53.43914715]]), array([[ -5.2939661 ,  12.50408565,  -3.3836204 ,  11.46347492,
#          -3.11700272],
#        [  3.33594549,   4.49276271, -21.27274803,   3.25830521,
#           2.72592104],
#        [ 12.1672839 ,   3.24528997,  -8.66299348,   3.72672717,
#           1.23120511],
#        [ 10.3940408 ,   3.30087455,  -4.01948676,   2.70480218,
#         -12.55941614],
#        [ -2.45839681,  11.65663521,  -4.12681375,  10.34695105,
#          -5.86721897]]), array([[42.10533155, 43.49564339, 52.02194957, 62.11164708, 40.55983008]])]
                                  
print(net_SGD.get_all_biases())

# [array([ -7.00500811, -34.05981431, -34.32622315, -30.94712279,
#          2.22703309]), array([-9.7092824 ,  7.25983628, -0.30632592, -0.8497914 , -4.67800886]), array([-79.72896034])]

