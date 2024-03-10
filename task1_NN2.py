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

def count_MSE(net, x_test, y_test, a=1, b=0):
    predictions = []
    for x in x_test:
        predictions.append(net.predict(x))
    predictions = np.array(predictions)
    predictions = predictions * a + b
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

#%% GD and SGD comparison

start = time.time()

f = [sigma, lambda x: x]
net_GD = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')
net_SGD = Net(n_neurons=[5, 1], n_inputs=1, functions=f, param_init='xavier')

epoch = 1
epochs = []
MSE_GD = []
MSE_SGD = []
current_MSE_SGD = math.inf
norms = [[] for _ in range(net_SGD.get_n_layers())]

while current_MSE_SGD > 4:
    epochs.append(epoch)
    epoch += 1
    net_GD.fit(x_train, (y_train - b) / a, batch_size=len(x_train), epochs=1, alpha=0.003)
    net_SGD.fit(x_train, (y_train - b) / a, batch_size=1, epochs=1, alpha=0.003)
    MSE_GD.append(count_MSE(net_GD, x_test, y_test, a, b))
    current_MSE_SGD = count_MSE(net_SGD, x_test, y_test, a, b)
    MSE_SGD.append(current_MSE_SGD)
    for norm, weights, biases in zip(norms, net_SGD.get_all_weights(), net_SGD.get_all_biases()):
        norm.append(np.linalg.norm(np.c_[weights, biases]))
        
end = time.time()

#%% results

plt.plot(epochs, MSE_GD, 'o')
plt.plot(epochs, MSE_SGD, 'o')
plt.legend(('GD', 'SGD'), loc='upper right')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

for norm in norms:
    plt.plot(epochs, norm, 'o')
plt.legend(('layer 1', 'layer 2'), loc='upper left')
plt.xlabel('epoch')
plt.ylabel('Frobenius norm')
plt.show()
    
plot_weights_on_layers(net_SGD)

predictions = []
for x in x_test:
    predictions.append(net_SGD.predict(x))
predictions = np.array(predictions)
predictions = predictions * a + b

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.legend(('test data', 'SGD prediction'), loc='upper left')
plt.show()

print("MSE on test set: ", current_MSE_SGD)
print("Last epoch: ", epoch - 1)
print("Time elapsed: ", end - start)

#%% weights

print(net_SGD.get_all_weights())

# [array([[-2.12139123],
#         [-2.91630697],
#         [-0.00601446],
#         [-0.72794638],
#         [-2.23782617]]), array([[ 3.97242838, -4.49332835,  1.60438546,  2.54970584, -2.18449294]])]

print(net_SGD.get_all_biases())

# [array([-2.96070156,  5.01817101,  0.35677676, -0.60697734,  1.39951975]), array([4.16702137])]


