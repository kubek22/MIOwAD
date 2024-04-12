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

df_train = read_csv("data/classification/rings3-balance-training.csv")
df_train.head()

xy_train = df_train[["x", "y"]].to_numpy()
c_train = df_train["c"].to_numpy()

df_test = read_csv("data/classification/rings3-balance-test.csv")
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

l = 3
k = 10
max_epochs = 1000
max_rises = 2
learning_rate = 0.003
threshold = 0.9

#%%

f = ['tanh', 'tanh', 'softmax']
net = Net(n_neurons=[k, k, l], n_inputs=2, functions=f, param_init='xavier', use_softmax=True)

train_scores = []
test_scores = []

warnings.filterwarnings('ignore')
start_time = time.time()

for i in range(max_epochs):
    net.fit(xy_train_scaled, c_train, batch_size=16, epochs=1, alpha=learning_rate, method='rmsprop')
    preds = predict(net, xy_train_scaled)
    classes = predict_class(preds)
    score = f1_score(c_train, classes, average='weighted')
    train_scores.append(score)
    
    preds = predict(net, xy_test_scaled)
    classes = predict_class(preds)
    score = f1_score(c_test, classes, average='weighted')
    test_scores.append(score)

    print("epoch: ", i + 1)
    print("train score : ", train_scores[-1])
    print("test score : ", test_scores[-1])
    print()
    
end_time = time.time()

#%% save results

save(train_scores, 'train_6_3_basic')
save(test_scores, 'test_6_3_basic')

#%%

train_scores = read('train_6_3_basic')
test_scores = read('test_6_3_basic')

epochs = np.arange(1, max_epochs + 1)
plt.plot(epochs, train_scores, 'o-', markersize=1)
plt.plot(epochs, test_scores, 'o-', markersize=1)
plt.legend(('train','test'), loc='lower right')
plt.xlabel('epoch')
plt.ylabel('F1 score')
plt.show()

#%% early stop l1
    
f = ['tanh', 'tanh', 'linear']
net = Net(n_neurons=[k, k, l], n_inputs=2, functions=f, param_init='xavier', use_softmax=True)

best_weights, best_biases, train_scores, test_scores = net.fit_until_rise(max_rises, threshold, xy_train_scaled, c_train, xy_test_scaled, c_test, scaler_y=None,
                   batch_size=16, epochs=max_epochs, alpha=learning_rate,
                   method='rmsprop', m_lambda=0.5, beta=0.9, 
                   regularization='l1', reg_lambda=0.8,
                   print_results=True)

#%% save results

save(train_scores, 'train_6_3_L1')
save(test_scores, 'test_6_3_L1')

#%%

train_scores = read('train_6_3_L1')
test_scores = read('test_6_3_L1')

epochs = np.arange(1, len(train_scores) + 1)
plt.plot(epochs, train_scores, 'o-', markersize=1)
plt.plot(epochs, test_scores, 'o-', markersize=1)
plt.legend(('train','test'), loc='lower right')
plt.xlabel('epoch')
plt.ylabel('F1 score')
plt.show()

#%% total results

test_scores1 = read('test_6_3_basic')
test_scores2 = read('test_6_3_L1')

epochs1 = np.arange(1, max_epochs + 1)
n = len(test_scores2)
epochs2 = np.arange(1, n + 1)

plt.plot(epochs1, test_scores1)
plt.plot(epochs2, test_scores2)
plt.legend(('basic','L1'), loc='lower right')
plt.xlabel('epoch')
plt.ylabel('test F1 score')
plt.show()


