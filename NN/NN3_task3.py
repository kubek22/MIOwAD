#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
import warnings
import copy

#%%

def save(array, file_name):
    file = open(file_name, 'wb')
    pickle.dump(array, file)
    file.close()

def read(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

#%%

df_train = read_csv("data/regression/multimodal-large-training.csv")
df_train.head()
x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/multimodal-large-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

#%%

plt.plot(x_train, y_train, 'o')
plt.show()

#%% data thinning

k = 10
x_train = x_train[::k]
y_train = y_train[::k]

plt.plot(x_train, y_train, 'o')
plt.show()

#%%

def ReLU(x):
    if x > 0:
        return x
    return 0.0

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

#%% scaling

scaler_x = MinMaxScaler(feature_range=(-0.9, 0.9))
x_train_scaled = scaler_x.fit_transform(np.transpose([x_train]))
x_test_scaled = scaler_x.transform(np.transpose([x_test]))

scaler_y = MinMaxScaler(feature_range=(-0.9, 0.9))
y_train_scaled = scaler_y.fit_transform(np.transpose([y_train]))
y_test_scaled = scaler_y.transform(np.transpose([y_test]))

plt.plot(x_train_scaled, y_train_scaled, 'o')
plt.plot(x_test_scaled, y_test_scaled, 'o')
plt.show()

#%%

warnings.filterwarnings('ignore') 

start = time.time()

f = [ReLU, ReLU, lambda x: x]
net_momentum = Net(n_neurons=[10, 10, 1], n_inputs=1, functions=f, param_init='xavier')
_w = copy.deepcopy(net_momentum.get_all_weights())
_b = copy.deepcopy(net_momentum.get_all_biases())
net_rmsprop = Net(weights=_w, biases=_b, functions=f)

epoch = 1
epochs = []
MSE_momentum = []
MSE_rmsprop = []
current_MSE = math.inf
e = 10

while current_MSE > 9:
    epochs.append(epoch)
    epoch += e
    net_momentum.fit(x_train_scaled, y_train_scaled, batch_size=1, epochs=e, alpha=0.003,
                     method='momentum', m_lambda=0.9)
    net_rmsprop.fit(x_train_scaled, y_train_scaled, batch_size=1, epochs=e, alpha=0.003,
                method='rmsprop', beta=0.9)
    mse_m = count_MSE(net_momentum, x_test_scaled, y_test, scaler_y)
    MSE_momentum.append(mse_m)
    mse_r = count_MSE(net_rmsprop, x_test_scaled, y_test, scaler_y)
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
plt.legend(('Momentum', 'RMSProp'), loc='upper left')
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
predictions = scaler_y.inverse_transform(np.array([predictions]))[0]
    
plt.plot(x_test, y_test, 'o', markersize=6)
plt.plot(x_test, predictions, 'o', markersize=3)
plt.legend(('test data', 'Momentum prediction'), loc='upper left')
plt.show()

print("Current epoch: ", epoch - 1)
print("MSE m: ", mse_m)
print("MSE r: ", mse_r)

#%% weights

print(net_momentum.get_all_weights())
# [array([[-0.5748361 ],
#        [ 2.64033955],
#        [ 1.00410411],
#        [ 5.04370787],
#        [ 0.32887136],
#        [ 1.37448763],
#        [ 1.65159127],
#        [-2.92409211],
#        [ 0.2754105 ],
#        [-2.60042263]]), array([[ 0.33519035,  0.15838879, -0.24255456, -0.71901888,  0.41947775,
#          0.06984221,  0.1408288 ,  0.08306617, -0.06561415, -1.72227561],
#        [ 0.3458716 , -1.3276101 , -0.38745166, -1.75265419, -0.17839509,
#          0.4982494 , -0.20094582, -0.784162  ,  0.25126537, -0.62506632],
#        [-0.56032867,  0.68246781, -0.46499685,  1.80688779, -0.6222469 ,
#         -0.59883974, -1.23734844, -0.20359636, -0.71362716, -0.26411145],
#        [-0.0457464 , -1.62819725, -0.3532288 , -2.44678859,  0.41996122,
#          0.62326481, -1.01935302, -2.43671383,  0.07177698, -0.52257195],
#        [-0.78459747, -1.29099722,  1.48682907, -2.88283209,  0.83658889,
#         -0.6089082 ,  2.52336486,  0.95179673,  0.2275996 , -1.01460782],
#        [ 0.27489926,  0.59504297, -0.16451525,  0.25551231, -0.62030479,
#          0.14350027, -0.34886162, -0.18636557,  0.15616345,  0.66014181],
#        [ 0.28287387,  0.76481081, -0.20603167,  2.09753683, -0.6980957 ,
#         -2.09560752, -0.04641816,  0.06759581, -0.77590987,  1.08808677],
#        [ 0.25266906,  0.5799558 , -0.63997032,  0.83653585, -0.43479384,
#         -0.45430222, -0.22476761, -0.18308199, -0.04431699,  0.89683146],
#        [-1.63954833, -0.97833964,  0.87296711, -2.25059529,  0.6048675 ,
#          0.50532115,  1.69641454, -0.21121771,  0.11861616, -0.13265908],
#        [ 0.09684525, -1.34064328,  1.29094735, -1.50470075,  0.33336277,
#         -0.72568343,  2.19409688,  0.1722145 ,  0.28861367, -2.64051565]]), array([[-0.86136765, -0.43853953, -0.6433709 , -0.98646675,  1.47955741,
#         -0.20828022,  0.73099183, -0.4061812 , -1.04775228, -0.94952869]])]

print(net_momentum.get_all_biases())
# [array([ 0.36380454, -0.41611203,  0.43227698, -1.04047604,  0.2756056 ,
#         0.88587569,  0.62643021, -2.26350939,  0.17557601, -0.10976818]), array([ 0.66985001,  1.11296018, -1.17483138,  1.54283895,  2.46570379,
#        -0.39370375, -0.96708311, -0.15161838,  1.67963087,  0.22687424]), array([0.74909801])]
