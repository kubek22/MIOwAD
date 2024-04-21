#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
import copy

#%%

def save(array, file_name):
    file= open(file_name, 'wb')
    pickle.dump(array, file)
    file.close()

def read(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

def sigma(x):
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

def ReLU(x):
    if x > 0:
        return x
    return 0.0

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

#%% scaling

scaler_x = MinMaxScaler(feature_range=(-0.8, 0.8))
x_train_scaled = scaler_x.fit_transform(np.transpose([x_train]))
x_test_scaled = scaler_x.transform(np.transpose([x_test]))

plt.plot(x_train_scaled, y_train, 'o')
plt.plot(x_test_scaled, y_test, 'o', markersize=2)
plt.show()

#%% comparison

start = time.time()

f = [sigma, sigma, lambda x: x]
net_momentum = Net(n_neurons=[5, 5, 1], n_inputs=1, functions=f, param_init='xavier')
_w = copy.deepcopy(net_momentum.get_all_weights())
_b = copy.deepcopy(net_momentum.get_all_biases())
net_rmsprop = Net(weights=_w, biases=_b, functions=f)

epoch = 1
epochs = []
MSE_momentum = []
MSE_rmsprop = []
current_MSE = math.inf
e = 10

while current_MSE > 3 or epoch > 1000:
    epochs.append(epoch)
    epoch += e
    net_momentum.fit(x_train_scaled, y_train, batch_size=1, epochs=e, alpha=0.0001,
                     method='momentum', m_lambda=0.9)
    net_rmsprop.fit(x_train_scaled, y_train, batch_size=1, epochs=e, alpha=0.003,
                method='rmsprop', beta=0.9)
    mse_m = count_MSE(net_momentum, x_test_scaled, y_test)
    MSE_momentum.append(mse_m)
    mse_r = count_MSE(net_rmsprop, x_test_scaled, y_test)
    MSE_rmsprop.append(mse_r)
    current_MSE = min(mse_m, mse_r)
    print("Current epoch: ", epoch - 1)
    print("MSE m: ", mse_m)
    print("MSE r: ", mse_r)
    print()
        
end = time.time()

#%% results

plt.plot(epochs, MSE_momentum, 'o-', markersize=4)
plt.plot(epochs, MSE_rmsprop, 'o-', markersize=4)
plt.legend(('Momentum', 'RMSprop'), loc='upper left')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

plt.plot(epochs, np.log(MSE_momentum), 'o-', markersize=4)
plt.plot(epochs, np.log(MSE_rmsprop), 'o-', markersize=4)
plt.legend(('Momentum', 'RMSProp'), loc='upper left')
plt.xlabel('epoch')
plt.ylabel('ln(MSE)')
plt.show()

predictions = []
for x in x_test_scaled:
    predictions.append(net_momentum.predict(x))
    
plt.plot(x_test, y_test, 'o', markersize=6)
plt.plot(x_test, predictions, 'o', markersize=3)
plt.legend(('test data', 'Momentum prediction'), loc='upper left')
plt.show()

predictions = []
for x in x_test_scaled:
    predictions.append(net_rmsprop.predict(x))
    
plt.plot(x_test, y_test, 'o', markersize=6)
plt.plot(x_test, predictions, 'o', markersize=3)
plt.legend(('test data', 'RMSProp prediction'), loc='upper left')
plt.show()

print("Current epoch: ", epoch - 1)
print("MSE m: ", mse_m)
print("MSE r: ", mse_r)

#%% momentum with decreasing learning rate (final model)

start = time.time()

f = [sigma, sigma, lambda x: x]
net_momentum = Net(n_neurons=[5, 5, 1], n_inputs=1, functions=f, param_init='xavier')

epoch = 1
epochs = []
MSE_momentum = []
mse_m = math.inf
e = 1
base_lr = 0.0001
lr = 0.0001

while mse_m > 3:
    epochs.append(epoch)
    epoch += e
    if mse_m < 100:
        lr = base_lr *math.sqrt(mse_m / epoch)
    net_momentum.fit(x_train_scaled, y_train, batch_size=1, epochs=e, alpha=lr,
                      method='momentum', m_lambda=0.9)
    mse_m = count_MSE(net_momentum, x_test_scaled, y_test)
    MSE_momentum.append(mse_m)
    print("Current epoch: ", epoch - 1)
    print("MSE m: ", mse_m)
    print()
    
end = time.time()

#%%

plt.plot(epochs, MSE_momentum, 'o-', markersize=4)
plt.legend(('Momentum', 'RMSprop'), loc='upper left')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

plt.plot(epochs, np.log(MSE_momentum), 'o-', markersize=4)
plt.legend(('Momentum', 'RMSProp'), loc='upper left')
plt.xlabel('epoch')
plt.ylabel('ln(MSE)')
plt.show()

predictions = []
for x in x_test_scaled:
    predictions.append(net_momentum.predict(x))
    
plt.plot(x_test, y_test, 'o', markersize=6)
plt.plot(x_test, predictions, 'o', markersize=3)
plt.legend(('test data', 'Momentum prediction'), loc='upper left')
plt.show()

print("Current epoch: ", epoch - 1)
print("MSE m: ", mse_m)

#%% weights

print(net_momentum.get_all_weights())
# [array([[ 123.12977134],
#        [ 256.01172546],
#        [-174.34386482],
#        [ 161.39987329],
#        [ -69.21665779]]), array([[ 21.39135067, -17.426634  , -19.70226152,  29.02103065,
#          -4.99537907],
#        [  3.36166536,  29.29766681,  -6.84394431,   4.05858177,
#         -11.98218306],
#        [ 17.55905107,  -4.51189133,  -4.96961447,  26.25726017,
#          -1.85301689],
#        [  2.27425446,  26.45895392, -14.83529618,   4.59774036,
#           3.61691083],
#        [  1.90272489,   3.8823038 , -24.71169442,   4.03156094,
#         -16.04850513]]), array([[37.76465922, 72.36072033, 42.19518269, 48.47383409, 39.2436571 ]])]

print(net_momentum.get_all_biases())
# [array([-70.39907854, -30.27127202, -60.8594334 , -92.81492787,
#        -24.33690631]), array([ -6.15155332,  -6.59668425, -11.89565752,   1.62577223,
#         10.85361893]), array([-80.08701915])]
