import numpy as np


class Neuron:
    def __init__(self, function, descendants=None, weights=None):
        self.function = function
        self.descendants = descendants  # list
        self.weights = weights  # incoming

    def set_weights(self, weights):
        self.weights = weights

    def compute(self, args):
        # use weights and given function to produce the output (and activate descendants to compute)
        return self.function(self.weights, args)


class Layer:
    def __init__(self, n_neurons, function, descendants, weights):
        self.n_neurons = n_neurons
        # initialize net
        self.neurons = [None for i in range(n_neurons)]
        for i in range(n_neurons):
            self.neurons[i] = Neuron(function, descendants, weights[i, :])

    def set_weights(self, weights):
        pass

    def compute(self, args):
        pass


class Net:
    # layers computing
    def __init__(self, neurons_in_layers, weights):
        self.n_layers = len(neurons_in_layers)
        self.neurons_in_layers = neurons_in_layers  # ex. [1, 2, 2]
        self.weights = weights
        # creating layers starting from bottom
