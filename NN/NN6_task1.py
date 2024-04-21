#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
import warnings

#%%

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

def save(array, file_name):
    file = open(file_name, 'wb')
    pickle.dump(array, file)
    file.close()

def read(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

#%%

df_train = read_csv("data/regression/multimodal-sparse-training.csv")
df_train.head()
x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/multimodal-sparse-test.csv")
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

#%% parameters

k = 10
max_epochs = 20000
max_rises = 100
learning_rate = 0.003
threshold = 500

#%% without regularization

f = ['tanh', 'tanh', 'linear']
net = Net(n_neurons=[k, k, 1], n_inputs=1, functions=f, param_init='xavier')
train_mse = []
test_mse = []

warnings.filterwarnings('ignore')
start_time = time.time()

for i in range(max_epochs):
    net.fit(x_train_scaled, y_train, batch_size=16, epochs=1, alpha=learning_rate, method='rmsprop')
    mse = count_MSE(net, x_train_scaled, y_train)
    train_mse.append(mse)
    mse = count_MSE(net, x_test_scaled, y_test)
    test_mse.append(mse)
    print("epoch: ", i + 1)
    print("mse train : ", train_mse[-1])
    print("mse test : ", test_mse[-1])
    print()
    
end_time = time.time()

#%% save results

save(train_mse, 'train_6_1_basic')
save(test_mse, 'test_6_1_basic')

#%%

train_mse = read('train_6_1_basic')
test_mse = read('test_6_1_basic')

epochs = np.arange(1, max_epochs + 1)

plt.plot(epochs, train_mse, 'o-', markersize=1)
plt.plot(epochs, test_mse, 'o-', markersize=1)
plt.legend(('train','test'), loc='upper right')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

#%% early stop l1
    
f = ['tanh', 'tanh', 'linear']
net = Net(n_neurons=[k, k, 1], n_inputs=1, functions=f, param_init='xavier')

best_weights, best_biases, train_mse, test_mse = net.fit_until_rise(max_rises, threshold, x_train, y_train, x_test, y_test, scaler_y=None,
                   batch_size=16, epochs=max_epochs, alpha=learning_rate,
                   method='rmsprop', m_lambda=0.5, beta=0.9, 
                   regularization='l1', reg_lambda=0.8,
                   print_results=False)

#%% save results

save(train_mse, 'train_6_1_L1')
save(test_mse, 'test_6_1_L1')

#%%

train_mse = read('train_6_1_L1')
test_mse = read('test_6_1_L1')

epochs = np.arange(1, len(train_mse) + 1)
plt.plot(epochs, train_mse, 'o-', markersize=1)
plt.plot(epochs, test_mse, 'o-', markersize=1)
plt.legend(('train','test'), loc='upper right')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

#%% total results

test_mse1 = read('test_6_1_basic')
test_mse2 = read('test_6_1_L1')

epochs1 = np.arange(1, max_epochs + 1)
n = len(test_mse2)
epochs2 = np.arange(1, n + 1)

plt.plot(epochs1, test_mse1)
plt.plot(epochs2, test_mse2)
plt.legend(('basic','L1'), loc='upper right')
plt.xlabel('epoch')
plt.ylabel('test MSE')
plt.show()
