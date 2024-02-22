from model import Net
import numpy as np

weights = np.array(
    [
        [[1, 1]],
        [[2, 2]],
        [[3, 3]]
    ]
)

functions = [lambda x: x]


net = Net(weights, functions)
