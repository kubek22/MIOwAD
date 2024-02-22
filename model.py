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

        n_neurons = self.get_n_neurons()

        self.function = function
        # possible
        self.functions = [function for i in range(n_neurons)]

    def compute(self, args):
        if self.weights.shape[1] != args.shape[0]:
            return None
        result = np.matmul(self.weights, args)
        #if type(self.functions) is list: ...
        return np.vectorize(self.function)(result)

    def get_n_neurons(self):
        return self.weights.shape[0]

    def get_weights(self):
        return self.weights

    def get_weight(self, neuron_index):
        return self.weights[neuron_index] if neuron_index < self.get_n_neurons() else None

    def get_functions(self):
        return self.functions

    def get_function(self, neuron_index):
        return self.functions[neuron_index] if neuron_index < self.get_n_neurons() else None

    def set_weights(self, weights):
        pass




class Net:
    # layers computing
    def __init__(self, weights, functions):
        """
        weights - list of matrices [layer, node, ancestor]
        functions - list of functions, one per layer
        """

        n_layers = weights.shape[0]

        if len(functions) == 1:
            functions = [functions for i in range(n_layers)]

        # creating layers
        self.layers = [Layer(functions[i], weights) for i in range(n_layers)]

    def compute(self, args):
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





