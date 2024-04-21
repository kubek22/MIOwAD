#%%
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
from numpy import array

#%%
df_train = read_csv("data/regression/square-simple-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]
y = x_train ** 2

df_test = read_csv("data/regression/square-simple-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

#%%
plt.plot(x_train, y_train, 'o')
plt.show()

#%%

def sigma(x):
    return 1 / (1 + math.e ** ((-1) * x))

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

#%%

k = -1 * 20

w = [array([[ 2 ],
        [ 1.20643329],
        [-1.3],
        [-1.3206462 ],
        [ 2.22886082]]) ,
 array([[-0.62166663, -0.92967308,  2.6523163 ,  1.25167433, -0.20654768],
        [ 2.11020537,  2.49005284, -4.62606786, -2.47856206, -4.61587503],
        [ 3.41791574,  0.63755489,  1.44823719,  2.12468459, -0.12109083],
        [-2.10857798, -2.84368046, -4.28747136, -2.59957875, -4.64830104],
        [ 2.33089549,  4.14200848,  4.56218429, -4.78558138,  2.61727907]]),
 array([[-0.01583961, -0.62838889,  0.05460575, -1.67955034,  0.34138198]]) * k]


# k = 2000

# w = [
#      np.transpose(array([[ 1.2268423 , -1.3768704 ,  0.31265107,  1.665269  , -3.610468  ]])),
# np.transpose(
# array([[-0.3428471 , -0.54420567,  0.12604073, -0.60694677, -0.5240614 ],
#        [ 0.600206  ,  0.4117785 ,  0.12047752,  0.5421165 ,  0.30355301],
#        [-0.44921818, -0.2553004 , -0.5932046 ,  0.44793484, -0.00824333],
#        [-0.54255277,  0.08376433, -0.29911622,  0.5032851 ,  0.27886987],
#        [-0.58762014, -0.0957372 , -0.5760486 ,  0.6942649 ,  0.950451  ]]) ),
# np.transpose(
# array([[ 0.75943714],
#        [ 0.87095636],
#        [ 1.02849   ],
#        [-0.607403  ],
#        [-0.34027073]]) ) 
# ]


biases = [0, 7, 0]

net = Net(w, [sigma, sigma, lambda x: x], biases)

predictions = []
for x in x_train:
    predictions.append(net.predict(x))
    
predictions
plt.plot(x_train, y, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y)
net.get_all_weights()

#%%

k = -1 * 20 * 90 + 1.5

w = [array([[ 1.88 ],
        [ 1.011],
        [-1.332],
        [-1.31],
        [ 2.31]]) ,
 array([[-0.1, -0.4,  2.5 ,  1, -0.1],
        [ 2.2,  2.5, -4.76, -2.5, -4.6],
        [ 3.5,  0.0,  -0.5,  2.3, 1],
        [-4, -1, -4, 0.5, -45],
        [ 2.6,  4.1,  4.55, -4.77,  2.6]]),
 array([[-0.016, -0.63,  0.0546, -1.8,  0.3414]]) * k]

w = [array([[ 1.84 ],
        [ 1.009],
        [-1.33],
        [-1.31],
        [ 2.3]]) ,
 array([[-0.01, -0.39,  2.7 ,  1, -0.15],
        [ 1.001,  3.895, -4.89, -2.47, -4.601],
        [ 3.55,  0.01,  -0.53,  2.28, 1],
        [-2.8, -1, -3.97, 0.56, -49],
        [ 2.71,  4.1,  4.54, -4.766,  2.64]]),
 array([[-0.01625, -0.6297,  0.05455, -1.87,  0.34144]]) * k]


biases = [0, 7 + 510 + -0.13, 0]

net = Net(w, [sigma, sigma, lambda x: x], biases)

predictions = []
for x in x_train:
    predictions.append(net.predict(x))
    
predictions
plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)

predictions_test = []
for x in x_test:
    predictions_test.append(net.predict(x))

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions_test, 'o')
plt.show()

MSE(predictions_test, y_test)


#%% good results

k = -1 * 20 * 90 + 3.8 

w = [array([[ 1.88 ],
        [ 1.011],
        [-1.332],
        [-1.31],
        [ 2.31]]) ,
 array([[-0.1, -0.4,  2.5 ,  1, -0.1],
        [ 2.2,  2.5, -4.76, -2.5, -4.6],
        [ 3.5,  0.0,  -0.5,  2.3, 1],
        [-4, -1, -4, 0.5, -45],
        [ 2.6,  4.1,  4.55, -4.77,  2.6]]),
 array([[-0.016, -0.63,  0.0546, -1.8,  0.3414]]) * k]

w = [array([[ 1.84 ],
        [ 1.009],
        [-1.33],
        [-1.31],
        [ 2.3]]) ,
 array([[-0.01, -0.39,  2.7 ,  1, -0.15],
        [ 1.001,  3.895, -4.89, -2.47, -4.601],
        [ 3.55,  0.01,  -0.53,  2.28, 1],
        [-2.8, -1, -3.97, 0.498, -43],
        [ 2.71,  4.1,  4.54, -4.766,  2.64]]),
 array([[-0.01625, -0.6297,  0.05455, -1.87,  0.34144]]) * k]


biases = [0, 7 + 510 + -0.13, 0]