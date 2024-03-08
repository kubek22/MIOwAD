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

df_train = read_csv("data/regression/multimodal-large-training.csv")
df_train.head()
x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/multimodal-large-test.csv")
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
plt.show()

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

f = [sigma, sigma, lambda x: x]
net = Net(n_neurons=[5, 5, 1], n_inputs=1, functions=f, param_init='xavier')
epoch = 0

#%%

start = time.time()

current_MSE = math.inf
while current_MSE > 10:
    net.fit(x_train_scaled, y_train_scaled, batch_size=16, epochs=1, alpha=0.003)
    epoch += 1
    current_MSE = count_MSE(net, x_test_scaled, y_test, scaler_y)
    print()
    print(current_MSE)
    print(epoch)
    
end = time.time()


#%%

predictions = []
for x in x_test_scaled:
    predictions.append(net.predict(x))
    
predictions = scaler_y.inverse_transform(np.array([predictions]))[0]

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

print(current_MSE)
print("Time elapsed: ", end - start)

#%%

save(net.get_all_weights(), "weights3.txt")
save(net.get_all_biases(), "biases3.txt")
