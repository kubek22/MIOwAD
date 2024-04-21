#%%
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math
 
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
plt.plot(x_train, y_train, 'o')
plt.show()

#%%
def sigma(x):
    return 1 / (1 + math.e ** ((-1) * x))

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

#%%
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'o')
plt.show()

#%%
k = 2

w = [
     [[2], [2], [2], [-2], [-2]],
     [[3 * k, 2 * k, 3 * k, 1 * k, 1.2 * k]]
     ]

biases = [-124, 0]

net1 = Net(w, [sigma, lambda x: x], biases)

net1.predict([1])
net1.predict(1)

# func = np.vectorize(net1.compute)
# np.array(x_train)
# func(np.array(x_train))

predictions = []
for x in x_train:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%% 1 layer (5)

# values = [i for i in range(-4, 0)] + [i for i in range(1, 5)]

import random

weights = None
min_MSE = math.inf
for i in range(10000):
    w = [
         [[random.random() * 10 - 5] for i in range(5)],
         [np.random.rand(1, 5)]
         ]
    biases = [-130, 0]
    net1 = Net(w, [sigma, lambda x: x], biases)
    predictions = []
    for x in x_test:
        predictions.append(net1.predict(x))
    mse = MSE(predictions, y_train)
    if mse < min_MSE:
        min_MSE = mse
        weights = net1.get_all_weights()
        
min_MSE
weights

net1 = Net(weights, [sigma, lambda x: x], biases)
predictions = []
for x in x_test:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%% 1 layer (10)

# values = [i for i in range(-4, 0)] + [i for i in range(1, 5)]

import random

weights = None
min_MSE = math.inf
for i in range(10000):
    w = [
         np.random.rand(10, 1) * 10 - 5,
         np.random.rand(1, 10) * 10 - 5
         ]
    biases = [-140, 0]
    net1 = Net(w, [sigma, lambda x: x], biases)
    predictions = []
    for x in x_test:
        predictions.append(net1.predict(x))
    mse = MSE(predictions, y_train)
    if mse < min_MSE:
        min_MSE = mse
        weights = net1.get_all_weights()
        
min_MSE
weights

net1 = Net(weights, [sigma, lambda x: x], biases)
predictions = []
for x in x_test:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%% 2 layers (5, 5)

# values = [i for i in range(-4, 0)] + [i for i in range(1, 5)]

import random

weights = None
min_MSE = math.inf
for i in range(1000):
    w = [
         np.random.rand(5, 1) * 10 - 5,
         np.random.rand(5, 5) * 10 - 5,
         np.random.rand(1, 5) * 10 - 5
         ]
    biases = [0, -150, 0]
    net1 = Net(w, [sigma, sigma, lambda x: x], biases)
    predictions = []
    for x in x_test:
        predictions.append(net1.predict(x))
    mse = MSE(predictions, y_train)
    if mse < min_MSE:
        min_MSE = mse
        weights = net1.get_all_weights()
        
min_MSE
weights

net1 = Net(weights, [sigma, sigma, lambda x: x], biases)
predictions = []
for x in x_test:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%%

k = 1

w = [
     [[-2], [-2], [1.2], [1.5], [1.5]],
     [np.array([1, 2, 2, 2, 2]) * k]
     ]

w = [[[-2],
        [ 1.5],
        [ 1.7],
        [-1.7],
        [-2.3]],
 np.array([[1.1, 4.7, 3.8, 2.5, 3.7]]) * k  ]

biases = [-150, 0]

net1 = Net(w, [sigma, lambda x: x], biases)

net1.predict([1])
net1.predict(1)

# func = np.vectorize(net1.compute)
# np.array(x_train)
# func(np.array(x_train))

predictions = []
for x in x_test:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

MSE(predictions, y_train)


#%%

k = 3

w = [
     [[-2.5], [0.5], [2], [0.5], [-2.5]],
     [[1 * k, 1 * k, 2.5 * k, 0.1 * k, 1 * k]]
     ]

biases = [-130, 0]

net1 = Net(w, [sigma, lambda x: x], biases)

net1.predict([1])
net1.predict(1)

# func = np.vectorize(net1.compute)
# np.array(x_train)
# func(np.array(x_train))

predictions = []
for x in x_test:
    predictions.append(net1.predict(x))
    
predictions
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%%
np.array(y_train)
np.sqrt(np.array(y_train))

x = np.array(x_train)
x = np.abs(x)
np.min(x)
np.where(x == np.min(x))
x[4]
y = np.array(y_train)
b = y[4]

y -= b
y

plt.plot(x, y, 'o')
plt.show()


a = np.mean((np.sqrt(y) / x) ** 2)
a # 88.81963626072975
b # -129.988852338856


#%%

w = [
     [[-2.5], [0.5], [2], [0.5], [-2.5]],
     [[1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, -1], [0, 1, 0, 1, 0], [1, 0, 0, 0, -1]],
     [[1, 0, 0, 0, 0]]
     ]

biases = [0, -130, 0]

net2 = Net(w, [sigma, sigma, lambda x: x], biases)

predictions = []
for x in x_train:
    predictions.append(net2.predict(x))
    
predictions
plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%%

w = [
     [[-2.5], [0.5], [2], [0.5], [-2.5]],
     [[1, 0, 0, 0, 0]]
     ]

#  biases = [0, -130, 0]

net2 = Net(w, [sigma, lambda x: x])

predictions = []
for x in x_train:
    predictions.append(net2.predict(x))
    
predictions
plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)

net2.get_n_layers()
net2.get_all_weights()
net2.get_n_inputs()

#%%

w = [
     [[-2.5], [0.5], [2], [0.5], [-2.5]],
     [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
     [[1, 0]]
     ]

#  biases = [0, -130, 0]

net2 = Net(w, [sigma, sigma, lambda x: x])

predictions = []
for x in x_train:
    predictions.append(net2.predict(x))
    
predictions
plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)


#%% after repairing sigma

k = 1

w = [
     [[-5], [-1], [-0.2], [3], [4]],
     [np.array([1, 1, 1, 1, 1]) * k]
     ]

w = [
     [[-1], [-2], [-1], [3], [4]],
     [np.array([1, 1, 1, 1, 1]) * k]
     ]


biases = [-130, 0]
biases = None

net = Net(w, [sigma, lambda x: x], biases)

predictions = []
for x in x_train:
    predictions.append(net.predict(x))
    
predictions
#plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%%

for i in range(1):
    k = 1

    w = [
         np.random.rand(5, 1) * 10 - 5,
         [np.array([1, 1, 1, 1, 1]) * k]
         ]


    biases = [-130, 0]
    biases = None

    net = Net(w, [sigma, lambda x: x], biases)

    predictions = []
    for x in x_train:
        predictions.append(net.predict(x))
        
    predictions
    #plt.plot(x_train, y_train, 'o')
    plt.plot(x_train, predictions, 'o')
    plt.show()

net.get_all_weights()
    
#%%

w1 = np.array([[ 3.24257939],
        [-4.84700725],
        [-1.93759815],
        [-2.58897785],
        [-4.11376312]])

k = 1

w = [
     w1,
     [np.array([1, 1, 1, 1, 1]) * k]
     ]



biases = [-130, 0]
biases = None

net = Net(w, [sigma, lambda x: x], biases)

predictions = []
for x in x_train:
    predictions.append(net.predict(x))
    
predictions
#plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%%

w1 = np.array([[ 3.24257939],
        [-4.84700725],
        [-1.93759815],
        [-2.58897785],
        [-4.11376312]])


w = [
     w1,
     [[1, 1, 1, 1, 1], [-1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 0]],
     [[1, 1, 1, 1, 1]]
     ]

#  biases = [0, -130, 0]

net = Net(w, [sigma, sigma, lambda x: x])

predictions = []
for x in x_train:
    predictions.append(net.predict(x))
    
predictions
#plt.plot(x_train, y_train, 'o')
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)

#%%

w = [
     np.random.rand(10, 1) * 10 - 5,
     np.random.rand(1, 10) * 4 - 2
     ]

#  biases = [0, -130, 0]

net = Net(w, [sigma, lambda x: x])

predictions = []
for x in x_train:
    predictions.append(net.predict(x))
    
predictions
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)
net.get_all_weights()
[np.array([[ 3.14257355],
        [ 4.59559132],
        [-0.37860804],
        [-3.45419699],
        [ 4.46500254],
        [-2.94542894],
        [-3.27604526],
        [ 4.53587543],
        [-4.92557584],
        [-4.04691855]]),
 np.array([[ 0.64662068, -1.24369241, -1.7270658 , -1.2958709 , -0.79709415,
         -1.71454966, -1.59942512, -1.75391343,  1.74718386, -1.11722554]])]


#%%

w = [
     np.random.rand(5, 1) * 10 - 5,
     np.random.rand(5, 5) * 10 - 5,
     np.random.rand(1, 5) * 4 - 2
     ]

#  biases = [0, -130, 0]

net = Net(w, [sigma, sigma, lambda x: x])

predictions = []
for x in x_train:
    predictions.append(net.predict(x))
    
predictions
plt.plot(x_train, predictions, 'o')
plt.show()

MSE(predictions, y_train)
net.get_all_weights()


#%%
from numpy import array

[array([[ 2.62951991],
        [-4.04363851],
        [ 1.45030064],
        [ 0.39881977],
        [ 2.6122984 ]]),
 array([[ 3.35565341,  3.43470372, -3.6224414 , -2.32346007,  0.34515143],
        [ 4.47631336,  0.0690766 ,  4.91495494, -0.95348373,  1.28022047],
        [-3.50610702,  2.45512387,  0.59853245,  2.22038426,  2.39115935],
        [ 4.61227037, -1.55986494,  2.94692257, -4.94380758, -4.34519933],
        [-1.41791184, -4.31959503, -3.59335553,  0.26713064,  2.0716831 ]]),
 array([[ 0.59747044,  1.39374384, -1.30345734, -1.8249042 , -0.94088215]])]


[array([[-3.44748142],
        [-2.40895961],
        [ 4.66892016],
        [ 4.41998419],
        [-4.47339721]]),
 array([[ 4.33708425, -0.23186898, -1.64503714, -1.32433478,  1.32747702],
        [ 0.57796652, -3.5989297 ,  1.43721999, -3.90349585, -1.26715384],
        [-2.29362836,  2.32375639,  2.75152643, -0.25856363,  3.84162536],
        [-3.61827485,  3.54010637,  0.21584891,  3.89692448, -2.37261046],
        [ 0.58542342,  3.25960616, -1.03428787, -4.88156872, -2.21793542]]),
 array([[-0.4848132 , -1.34179268, -0.44464387,  0.23394765,  1.0443377 ]])]


[array([[-4.76809497],
        [-0.19398047],
        [ 2.3793999 ],
        [-0.7393048 ],
        [-1.81285256]]),
 array([[ 3.87674799, -0.94618396, -4.53727456,  4.61860798,  4.38687294],
        [ 4.61021639, -0.33857176, -2.89850816, -2.75465985,  2.2530155 ],
        [ 4.36131285,  1.07316228,  0.30824517, -3.33448143, -2.72294797],
        [ 0.61997399, -0.7411127 ,  4.3612591 , -0.23641337, -2.59743258],
        [-3.13381738,  1.71020199,  2.52691913,  1.99820676,  3.66845775]]),
 array([[-1.79444015,  0.68664533, -0.32756472, -1.44719848, -1.41075288]])]


[array([[ 1.2495804 ],
        [ 1.20643329],
        [-0.76025944],
        [-1.3206462 ],
        [ 2.22886082]]),
 array([[-0.62166663, -0.92967308,  2.6523163 ,  1.25167433, -0.20654768],
        [ 2.11020537,  2.49005284, -4.62606786, -2.47856206, -4.61587503],
        [ 3.41791574,  0.63755489,  1.44823719,  2.12468459, -0.12109083],
        [-2.10857798, -2.84368046, -4.28747136, -2.59957875, -4.64830104],
        [ 2.33089549,  4.14200848,  4.56218429, -4.78558138,  2.61727907]]),
 array([[-0.01583961, -0.62838889,  0.05460575, -1.67955034,  0.34138198]])]



[array([[ 0.35942594],
        [-3.41679044],
        [ 1.14581865],
        [-3.85146514],
        [-2.67932719]]),
 array([[-1.8470575 ,  3.38660671,  2.03832142, -2.65587893, -3.02999542],
        [-2.59771367, -3.22659098,  3.13111585,  2.42465615,  4.33092421],
        [-0.23311317, -4.46835957, -2.74610687,  4.1107249 ,  1.71155158],
        [ 3.91662389, -1.58141277, -1.71660148, -1.7838581 , -3.91143077],
        [-4.21204419,  4.02875459, -4.19800944, -0.07121988,  3.29063716]]),
 array([[0.49400081, 0.82702109, 0.65346169, 0.61399661, 0.1681236 ]])]


#%%
from numpy import array
# transpose

[
array([[ 1.2268423 , -1.3768704 ,  0.31265107,  1.665269  , -3.610468  ]]),

array([[-0.3428471 , -0.54420567,  0.12604073, -0.60694677, -0.5240614 ],
       [ 0.600206  ,  0.4117785 ,  0.12047752,  0.5421165 ,  0.30355301],
       [-0.44921818, -0.2553004 , -0.5932046 ,  0.44793484, -0.00824333],
       [-0.54255277,  0.08376433, -0.29911622,  0.5032851 ,  0.27886987],
       [-0.58762014, -0.0957372 , -0.5760486 ,  0.6942649 ,  0.950451  ]]),

array([[ 0.75943714],
       [ 0.87095636],
       [ 1.02849   ],
       [-0.607403  ],
       [-0.34027073]])]
