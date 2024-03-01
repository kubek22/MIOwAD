#%%

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from model import Net
import math

#%%

def sigma(x):
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

#%%

f = [lambda x: x, lambda x: x]

net = Net(functions=f, n_neurons=[5, 3], n_inputs=2)

net.get_all_functions()
net.get_all_biases()
w=net.get_all_weights()

net.predict([1, 2])


np.array([4, 6]) * np.array([2, 3]) + np.array([1, 1]) * 2

#%%

f = [lambda x: x, lambda x: x]

net = Net(functions=f, n_neurons=[5, 3], n_inputs=2, param_init="xavier")

net.get_all_functions()
net.get_all_biases()
net.get_all_weights()

net.predict([1, 2])

#%%

scales = np.array([1, 1])
shifts = - scales
scales *= 2


