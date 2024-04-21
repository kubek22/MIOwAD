#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
import pickle
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

conversion = {True: 1, False: 0}

df_train = read_csv("data/classification/easy-training.csv")
df_train.head()

xy_train = df_train[["x", "y"]].to_numpy()
c_train = df_train["c"].map(conversion).to_numpy()

df_test = read_csv("data/classification/easy-test.csv")
df_test.head()

xy_test = df_test[["x", "y"]].to_numpy()
c_test= df_test["c"].map(conversion).to_numpy()

#%%
        
plt.scatter(xy_train[:,0], xy_train[:,1], c=(c_train+0.5)/2)
plt.show()

#%%

f = [sigma, "softmax"]
net_softmax = Net(n_neurons=[3, 2], n_inputs=2, functions=f, param_init='xavier', use_softmax=True)
f = [sigma, lambda x: x]
net_basic = Net(n_neurons=[3, 2], n_inputs=2, functions=f, param_init='xavier', classification=True)

#%%

epoch = 1
score = 0
scores_softmax = []
scores_basic = []
epochs = []
while score < 0.99:
    epochs.append(epoch)
    epoch += 1
    net_softmax.fit(xy_train, c_train, batch_size=1, epochs=1, alpha=0.003,
                      method='momentum', m_lambda=0.9)
    net_basic.fit(xy_train, c_train, batch_size=1, epochs=1, alpha=0.003,
                      method='momentum', m_lambda=0.9)
    preds = predict(net_softmax, xy_test)
    classes = predict_class(preds)
    score = f1_score(c_test, classes)
    scores_softmax.append(score)
    preds = predict(net_basic, xy_test)
    classes = predict_class(preds)
    scores_basic.append(f1_score(c_test, classes))
    
#%%

plt.plot(epochs, scores_softmax, 'o-')
plt.plot(epochs, scores_basic, 'o-')
plt.legend(('softmax', 'basic'), loc='upper left')
plt.xticks(np.arange(min(epochs), max(epochs) + 1, 1.0))
plt.xlabel('epoch')
plt.ylabel('F1 score')
plt.show()

print('Current epoch: ', epoch - 1)
print('F1 score softmax: ', scores_softmax[-1])
print('F1 score basic: ', scores_basic[-1])

#%%

print(net_softmax.get_all_weights())
# [array([[-0.4828805 ,  0.39143151],
#         [ 0.38515477, -0.39330314],
#         [-0.72477795,  0.72480169]]),
#  array([[-0.47592309, -1.05047658,  1.09867105],
#         [-1.64283493,  0.4662828 , -0.7698204 ]])]

print(net_softmax.get_all_biases())
# [array([0.00760212, 0.92893185, 0.19245558]), array([0.15982369, 0.80760132])]

