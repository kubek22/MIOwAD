import numpy as np


class Layer:
    def __init__(self, function, weights):
        """
        weights as matrix
        one function for all layer
        """

        if type(weights) is list:
            weights = np.array(weights)

        self.weights = weights
        self.n_neurons = self.weights.shape[0]
        self.function = function
        # possible
        self.functions = [function for i in range(self.n_neurons)]

    def compute(self, args):
        if type(args) is list:
            args = np.array(args)
        if self.weights.shape[1] != args.shape[0]:
            return None
        result = np.matmul(self.weights, args)
        # if type(self.functions) is list: ...
        f = np.vectorize(self.function)
        return f(result)

    def get_n_neurons(self):
        return self.n_neurons

    def get_weights(self):
        return self.weights

    def get_weight(self, neuron_index):
        return self.weights[neuron_index] if neuron_index < self.get_n_neurons() else None

    def get_functions(self):
        return self.functions

    def get_function(self, neuron_index):
        return self.functions[neuron_index] if neuron_index < self.get_n_neurons() else None

    def get_n_inputs(self):
        return self.weights.shape[1]

    def set_function(self, function):
        self.function = function

    def set_weights(self, weights):
        if weights.shape[0] != self.get_n_neurons():
            return None
        self.weights = weights

    def are_next_weights_correct(self, weights):
        if weights.shape[1] != self.get_n_neurons():
            return False
        return True

    def set_neuron_weights(self, neuron_index, weights):
        if self.weights.shape[1] != weights.shape[0]:
            return None
        if not neuron_index < self.get_n_neurons():
            return None
        self.weights[neuron_index, :] = weights

    def change_neuron_weight(self, neuron_index, ancestor_index, weight):
        if not neuron_index < self.get_n_neurons():
            return None
        if not ancestor_index < self.weights.shape[1]:
            return None
        self.weights[neuron_index, ancestor_index] = weight

    def set_n_neurons(self, n_neurons):
        self.n_neurons = n_neurons


class Net:
    def __init__(self, weights, functions):
        """
        weights - list of matrices [layer, node, ancestor]
        functions - list of functions, one per layer
        """

        n_layers = weights.shape[0]

        if len(functions) == 1 and n_layers > 1:
            functions = [functions for i in range(n_layers)]

        # creating layers
        self.layers = [Layer(functions[i], weights[i]) for i in range(n_layers)]

    def compute(self, args):
        """returns None in case of dimension problems"""
        for layer in self.layers:
            args = layer.compute(args)
            if args is None:
                return None
        return args

    def get_n_layers(self):
        return len(self.layers)

    def get_all_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def get_layer_weights(self, layer_index):
        return self.layers[layer_index] if layer_index < self.get_n_layers() else None

    def get_weight(self, layer_index, neuron_index):
        return self.layers[layer_index].get_weight(neuron_index) if layer_index < self.get_n_layers() else None

    def get_all_functions(self):
        return [layer.get_functions() for layer in self.layers]

    def get_layer_functions(self, layer_index):
        return self.layers[layer_index].get_functions() if layer_index < self.get_n_layers() else None

    def get_function(self, layer_index, neuron_index):
        return self.layers[layer_index].get_function(neuron_index) if layer_index < self.get_n_layers() else None

    def get_n_inputs(self):
        return self.layers[0].get_n_inputs()

    def set_all_functions(self, functions):
        if len(functions) != self.get_n_layers():
            return None
        for layer, function in zip(self.layers, functions):
            layer.set_function(function)

    def set_layer_function(self, layer_index, function):
        self.layers[layer_index].set_function(function) if layer_index < self.get_n_layers() else None

    def set_layer_weights(self, layer_index, weights):
        if not layer_index < self.get_n_layers():
            return None

        if type(weights) is list:
            weights = np.array(weights)

        layer = self.layers[layer_index]
        if layer_index == 0:
            # first layer
            layer.set_weights(weights)
            return
        prev_layer = self.layers[layer_index - 1]
        if not prev_layer.are_next_weights_correct(weights):
            return None
        layer.set_weights(weights)

    def set_neuron_weights(self, layer_index, neuron_index, weights):
        if type(weights) is list:
            weights = np.array(weights)
        if layer_index < self.get_n_layers():
            self.layers[layer_index].set_neuron_weights(neuron_index, weights)

    def change_neuron_weight(self, layer_index, neuron_index, ancestor_index, weight):
        if layer_index < self.get_n_layers():
            return None
        self.layers[layer_index].change_neuron_weight(neuron_index, ancestor_index, weight)

    def change_neurons_number(self, layer_index, n_neurons, weights, next_weights):
        if layer_index < self.get_n_layers():
            return None
        if n_neurons < 1:
            return None
        self.set_layer_weights(layer_index, weights)
        self.layers[layer_index].set_n_neurons(n_neurons)
        if layer_index == self.get_n_layers() - 1:
            return
        self.layers[layer_index + 1].set_weights(next_weights)


