import numpy as np
import math


class SOM:
    def __init__(self, n_rows, n_cols, n_inputs):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_neurons = n_rows * n_cols
        self.n_inputs = n_inputs
        self.weights = np.random.rand(self.n_neurons, n_inputs)
        self.mesh = RectangularMesh(n_rows, n_cols)
        self.neighborhood_function = self.gaussian_function
        
    def get_distances(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=1))
        
    def get_nearest_index(self, x):
        return np.argmin(self.get_distances(self.weights, x))
    
    def gaussian_function(self, x, epoch, scale):
        return np.exp(-(x * epoch * scale) ** 2)
    
    def mexican_hat(self, x, epoch, scale):
        x = x * epoch * scale
        return (2 - 4 * x ** 2) * np.exp(-x ** 2)
    
    def fit(self, data, epochs, init_lr=0.003, shuffle=True, scale=1):
        data = data.copy()
        for epoch in range(1, epochs + 1):
            np.random.shuffle(data)
            for x in data:
                winner_index = self.get_nearest_index(x)
                indices = self.mesh.get_indices()
                positions = self.mesh.get_positions_on_mesh(indices)
                winner_position = positions[winner_index]
                distances = self.get_distances(positions, winner_position)
                neighborhood = self.neighborhood_function(distances, epoch, scale)
                neighborhood = neighborhood[:, np.newaxis]
                lr = init_lr * math.e ** (-epoch/epochs)
                self.weights += lr * neighborhood * (x - self.weights)
        
class Mesh:
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
    
    def get_index(self, index):
        row = index // self.n_cols
        column = index - row * self.n_cols
        return np.array([row, column])
    
    def get_indices(self, indices=None):
        if indices is None:
            indices = np.arange(0, self.n_rows * self.n_cols)
        positions = []
        for index in indices:
            positions.append(self.get_index(index))
        return np.array(positions)
    
    def get_position_on_mesh(self, indices):
        pass
    
    def get_positions_on_mesh(self, indices):
        res = []
        for index in indices:
            res.append(self.get_position_on_mesh(index))
        return np.array(res)

class RectangularMesh(Mesh):
    def get_position_on_mesh(self, index):
        return np.array(index)
    
    def get_positions_on_mesh(self, indices):
        return indices

class HexagonalMesh(Mesh):
    pass

