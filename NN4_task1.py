#%% 

import numpy as np
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
import warnings 
from sklearn.metrics import f1_score

#%%

def save(array, file_name):
    file = open(file_name, 'wb')
    pickle.dump(array, file)
    file.close()

def read(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

def ReLU(x):
    if x > 0:
        return x
    return 0.0

def sigma(x):
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

def predict_class(predictions):
    classes = []
    for p in predictions:
        classes.append(np.where(np.max(p) == p))
    return np.array(classes).reshape(-1)

def predict(net, x_data):
    predictions = []
    for x in x_data:
        predictions.append(net.predict(x))
    return np.array(predictions)

#%%

df_train = read_csv("data/classification/rings3-regular-training.csv")
df_train.head()

xy_train = df_train[["x", "y"]].to_numpy()
c_train = df_train["c"].to_numpy()

df_test = read_csv("data/classification/rings3-regular-test.csv")
df_test.head()

xy_test = df_test[["x", "y"]].to_numpy()
c_test= df_test["c"].to_numpy()

#%%
        
plt.scatter(xy_train[:,0], xy_train[:,1], c=(c_train+0.5)/2)
plt.show()

#%%

plt.scatter(xy_test[:,0], xy_test[:,1], c=(c_test+0.5)/2)
plt.show()

#%% scaling

scaler_xy = MinMaxScaler(feature_range=(-0.9, 0.9))
xy_train_scaled = scaler_xy.fit_transform(xy_train)
xy_test_scaled = scaler_xy.transform(xy_test)

#%%

f = [sigma, sigma, "softmax"]
net = Net(n_neurons=[20, 20, 3], n_inputs=2, functions=f, param_init='xavier', use_softmax=True)

#%%

net.fit(xy_train_scaled, c_train, batch_size=1, epochs=100, alpha=0.005,
                  method='rmsprop', m_lambda=0.9)

#%%

preds = predict(net, xy_test_scaled)
classes = predict_class(preds)

plt.scatter(xy_test[:,0], xy_test[:,1], c=(classes+0.5)/2)
plt.show()

# f1_score(c_test, classes, average=None)
f1_score(c_test, classes, average='micro')
f1_score(c_test, classes, average='macro')
f1_score(c_test, classes, average='weighted')

#%%


