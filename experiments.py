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

res = net.compute(np.array([1, 2]))
print(res)


