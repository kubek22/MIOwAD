#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import warnings

#%%

def save(array, file_name):
    file = open(file_name, 'wb')
    pickle.dump(array, file)
    file.close()

def read(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

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

#%% scaling

scaler_xy = MinMaxScaler(feature_range=(-0.9, 0.9))
xy_train_scaled = scaler_xy.fit_transform(xy_train)
xy_test_scaled = scaler_xy.transform(xy_test)

#%% parameters

k = 5
l = 3
max_epochs = 1000

#%% two layer tanh vs. three layer relu

f = ['tanh', 'tanh', 'softmax']
net_tanh = Net(n_neurons=[k, k, l], n_inputs=2, functions=f, param_init='xavier', use_softmax=True)

f = ['relu', 'relu', 'relu', 'softmax']
net_relu = Net(n_neurons=[k, k, k, l], n_inputs=2, functions=f, param_init='xavier', use_softmax=True)


scores_tanh = []
scores_relu = []

warnings.filterwarnings('ignore') 
start_time = time.time()

for i in range(max_epochs):
    net_tanh.fit(xy_train_scaled, c_train, batch_size=1, epochs=1, alpha=0.003, method='rmsprop')
    net_relu.fit(xy_train_scaled, c_train, batch_size=1, epochs=1, alpha=0.003, method='rmsprop')

    preds = predict(net_tanh, xy_test_scaled)
    classes = predict_class(preds)
    score_tanh = f1_score(c_test, classes, average='weighted')
    scores_tanh.append(score_tanh)
    
    preds = predict(net_relu, xy_test_scaled)
    classes = predict_class(preds)
    score_relu = f1_score(c_test, classes, average='weighted')
    scores_relu.append(score_relu)

    print("epoch: ", i + 1)
    print("score tanh : ", score_tanh)
    print("score relu : ", score_relu)
    print()
    
end_time = time.time()

#%% save results

save(scores_tanh, 'class1_tanh2')
save(scores_relu, 'class1_relu3')

#%% summary

scores_tanh = read('class1_tanh2')
scores_relu = read('class1_relu3')
epochs = np.arange(1, max_epochs + 1)

plt.plot(epochs, scores_tanh, linewidth=0.5)
plt.plot(epochs, scores_relu, linewidth=0.5)
plt.legend(('tanh','ReLU'), loc='lower right')
plt.xlabel('epoch')
plt.ylabel('F1 score')
plt.show()  

print('max tanh score: ', max(scores_tanh))
print('max ReLU score: ', max(scores_relu))

#%%

df_train = read_csv("data/classification/rings5-regular-training.csv")
df_train.head()

xy_train = df_train[["x", "y"]].to_numpy()
c_train = df_train["c"].to_numpy()

df_test = read_csv("data/classification/rings5-regular-test.csv")
df_test.head()

xy_test = df_test[["x", "y"]].to_numpy()
c_test= df_test["c"].to_numpy()

#%%
        
plt.scatter(xy_train[:,0], xy_train[:,1], c=(c_train+0.5)/2)
plt.show()

#%% scaling

scaler_xy = MinMaxScaler(feature_range=(-0.9, 0.9))
xy_train_scaled = scaler_xy.fit_transform(xy_train)
xy_test_scaled = scaler_xy.transform(xy_test)

#%% parameters

k = 5
l = 5
max_epochs = 1000

#%% two layer tanh vs. three layer relu

f = ['tanh', 'tanh', 'softmax']
net_tanh = Net(n_neurons=[k, k, l], n_inputs=2, functions=f, param_init='xavier', use_softmax=True)

f = ['relu', 'relu', 'relu', 'softmax']
net_relu = Net(n_neurons=[k, k, k, l], n_inputs=2, functions=f, param_init='xavier', use_softmax=True)


scores_tanh = []
scores_relu = []

warnings.filterwarnings('ignore') 
start_time = time.time()

for i in range(max_epochs):
    net_tanh.fit(xy_train_scaled, c_train, batch_size=1, epochs=1, alpha=0.003, method='rmsprop')
    net_relu.fit(xy_train_scaled, c_train, batch_size=1, epochs=1, alpha=0.003, method='rmsprop')

    preds = predict(net_tanh, xy_test_scaled)
    classes = predict_class(preds)
    score_tanh = f1_score(c_test, classes, average='weighted')
    scores_tanh.append(score_tanh)
    
    preds = predict(net_relu, xy_test_scaled)
    classes = predict_class(preds)
    score_relu = f1_score(c_test, classes, average='weighted')
    scores_relu.append(score_relu)

    print("epoch: ", i + 1)
    print("score tanh : ", score_tanh)
    print("score relu : ", score_relu)
    print()
    
end_time = time.time()

#%% save results

save(scores_tanh, 'class2_tanh2')
save(scores_relu, 'class2_relu3')

#%% summary

scores_tanh = read('class2_tanh2')
scores_relu = read('class2_relu3')
epochs = np.arange(1, max_epochs + 1)

plt.plot(epochs, scores_tanh, linewidth=0.5)
plt.plot(epochs, scores_relu, linewidth=0.5)
plt.legend(('tanh','ReLU'), loc='lower right')
plt.xlabel('epoch')
plt.ylabel('F1 score')
plt.show()  

print('max tanh score: ', max(scores_tanh))
print('max ReLU score: ', max(scores_relu))


