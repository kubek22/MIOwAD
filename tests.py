from model import Net
import numpy as np

weights = np.array(
    [
        [[1, 1],
        [2, 2],
        [3, 3]]
    ]
)

print(weights[0, :])
print(weights.shape)
print(weights.shape[0])

functions = [lambda x: x]


net = Net(weights, functions)

print(net.get_n_layers())
print(net.get_all_weights())
print(net.get_n_inputs())
print(net.get_all_functions())

if type([1, 2]) is list:
    print(np.array([1, 2]))
    print(type(np.array([1, 2])))

res = net.predict(np.array([1, 2]))
print(res)

#%%

w = [
    [[1,3], [4,5]],
    [[2,1]]
    ]

f = [lambda x: 2*x, lambda x: x/8]

net2 = Net(w, f)

net2.predict([1, 2])

net2.get_all_functions()
net2.get_all_weights()
net2.get_function(0, 0)
net2.get_function(0, 1)
net2.get_function(0, 2)

net2.get_function(1, 0)
net2.get_function(1, 1)

net2.get_layer_functions(0)
net2.get_layer_functions(1)
net2.get_layer_functions(2)

net2.get_layer_weights(0)
net2.get_layer_weights(1)
net2.get_layer_weights(2)

net2.get_n_inputs()
net2.get_n_layers()

net2.get_weight(0, 0)
net2.get_weight(0, 1)
net2.get_weight(0, 2)
net2.get_weight(0, -1)
net2.get_weight(1, 0)
net2.get_weight(-1, 0)

net2.set_all_functions([lambda x: x] * 2)
net2.predict([1, 2])

net2.set_all_functions(f)
net2.predict([1, 2])

net2.set_layer_function(1, lambda x: x)
net2.predict([1, 2])

net2.set_layer_function(2, lambda x: x)
net2.predict([1, 2])

net2.set_layer_weights(1, [[1,1]])
net2.predict([1, 2])

net2.set_layer_weights(1, [[1,1,1]])
net2.predict([1, 2])
net2.get_all_weights()

net2.set_layer_weights(1, [[1,1], [1,1]])
print(net2.set_layer_weights(1, [[1,1], [1,1]]))
net2.get_all_weights()

net2.set_neuron_weights(1, 0, [2,1])
net2.predict([1, 2])
net2.get_all_weights()

net2.set_neuron_weight(1, 0, 0, 1)
net2.predict([1, 2])
net2.get_all_weights()

net2.set_neurons_number(1, 2, [[1, 1], [1, 2]])
net2.get_all_weights()
net2.predict([1, 2])

net2.set_neuron_weights(1, 1, [2,1])
net2.get_all_weights()
net2.predict([1, 2])


net2.set_neurons_number(0, 1, [[1, 1]])
net2.get_all_weights()

net2.set_neurons_number(0, 1, [[1, 1]], [[1], [1]])
net2.get_all_weights()
net2.predict([1, 2])

net2.set_all_functions([lambda x: x] * 2)
net2.predict([1, 2])

net2.set_neurons_number(4, 1, [[1, 1]], [[1], [1]])

