import numpy as np
import math


class Net:
    class Layer:
        def __init__(self, function, weights, bias=0):
            """
            weights as matrix
            one function for all layer
            """

            if type(weights) is list:
                weights = np.array(weights)
            self.weights = weights
            self.n_neurons = self.weights.shape[0]
            self.function = function
            # possible (not needed)
            # self.functions = [function for _ in range(self.n_neurons)]
            if type(bias) is list:
                bias = np.array(bias)
            self.bias = bias
            self.__args = None

        def compute(self, args):
            if type(args) is int or type(args) is float:
                args = [args]
            if type(args) is list:
                args = np.array(args)
            if self.weights.shape[1] != args.shape[0]:
                return None
            self.__args = args
            result = np.matmul(self.weights, args)
            result = result + self.bias
            # if type(self.functions) is list: ...
            f = np.vectorize(self.function)
            result = f(result)
            return result

        def get_n_neurons(self):
            return self.n_neurons

        def get_weights(self):
            return self.weights

        def get_weight(self, neuron_index):
            return self.weights[neuron_index] if neuron_index < self.get_n_neurons() else None

        def get_function(self):
            return self.function

        # def get_functions(self):
        #     return self.functions

        # def get_function(self, neuron_index):
        #     return self.functions[neuron_index] if neuron_index < self.get_n_neurons() else None

        def get_n_inputs(self):
            return self.weights.shape[1]

        def get_bias(self):
            return self.bias

        def set_function(self, function):
            self.function = function

        def set_bias(self, bias):
            if type(bias) is list:
                bias = np.array(bias)
            self.bias = bias

        def set_weights(self, weights):
            if weights.shape[0] != self.get_n_neurons():
                raise ValueError('Wrong weights dimension')
            self.weights = weights

        def are_next_weights_correct(self, weights):
            if weights.shape[1] != self.get_n_neurons():
                return False
            return True

        def set_neuron_weights(self, neuron_index, weights):
            if self.weights.shape[1] != weights.shape[0]:
                raise ValueError('Wrong weights length')
            if not neuron_index < self.get_n_neurons():
                raise IndexError('Wrong neuron index')
            self.weights[neuron_index, :] = weights

        def set_neuron_weight(self, neuron_index, ancestor_index, weight):
            if not neuron_index < self.get_n_neurons():
                raise IndexError('Wrong neuron index')
            if not ancestor_index < self.weights.shape[1]:
                raise IndexError('Wrong ancestor index')
            self.weights[neuron_index, ancestor_index] = weight

        def set_n_neurons(self, n_neurons):
            self.n_neurons = n_neurons

    def __init__(self, weights=None, functions=None, biases=None, n_neurons=None, n_inputs=None, param_init=None):
        """
        n_neurons - list with numbers of neurons
        weights - list of matrices [layer, node, ancestor]
        functions - list of functions, one per layer
        """
        if functions is None:
            raise ValueError('Functions parameter missing')
        if weights is not None:
            self.__initialize(weights, functions, biases)
            return
        if n_neurons is None or n_inputs is None:
            raise ValueError('Wrong arguments given')
        if param_init is None:
            weights, biases = self.__random_weights(n_neurons, n_inputs)
        if param_init == 'xavier':
            weights, biases = self.__xavier_weights(n_neurons, n_inputs)
        self.__initialize(weights, functions, biases)

    def __initialize(self, weights, functions, biases=None):
        n_layers = len(weights)
        if biases is None:
            biases = [0 for _ in range(n_layers)]

        if len(functions) == 1 and n_layers > 1:
            functions = [functions for _ in range(n_layers)]
        self.layers = [self.Layer(functions[i], weights[i], biases[i]) for i in range(n_layers)]

    def __random_weights(self, n_neurons, n_inputs, scales=None, shifts=None):
        n_layers = len(n_neurons)
        weights = [None for _ in range(n_layers)]
        weights[0] = np.random.rand(n_neurons[0], n_inputs)
        for i in range(1, n_layers):
            weights[i] = np.random.rand(n_neurons[i], n_neurons[i - 1])
        biases = [np.random.rand(n) for n in n_neurons]
        if scales is not None and shifts is not None:
            for i in range(n_layers):
                weights[i] = weights[i] * scales[i] + shifts[i]
        return weights, biases

    def __xavier_weights(self, n_neurons, n_inputs):
        n_layers = len(n_neurons)
        scales = np.ones(n_layers)
        scales[0] /= math.sqrt(n_inputs + n_neurons[0])
        for i in range(1, n_layers):
            scales[i] /= math.sqrt(n_neurons[i - 1] + n_neurons[i])
        scales *= math.sqrt(6)
        shifts = - scales
        scales *= 2
        return self.__random_weights(n_neurons, n_inputs, scales, shifts)

    def predict(self, args):
        for layer in self.layers:
            args = layer.compute(args)
            if args is None:
                raise Exception
        if len(args) == 1:
            args = args[0]
        return args

    def get_n_layers(self):
        return len(self.layers)

    def get_all_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def get_layer_weights(self, layer_index):
        return self.layers[layer_index].get_weights() if layer_index < self.get_n_layers() else None

    def get_weight(self, layer_index, neuron_index):
        return self.layers[layer_index].get_weight(neuron_index) if layer_index < self.get_n_layers() else None

    def get_all_functions(self):
        return [layer.get_function() for layer in self.layers]

    def get_layer_function(self, layer_index):
        return self.layers[layer_index].get_function() if layer_index < self.get_n_layers() else None

    # def get_function(self, layer_index, neuron_index):
    #     return self.layers[layer_index].get_function(neuron_index) if layer_index < self.get_n_layers() else None

    def get_n_inputs(self):
        return self.layers[0].get_n_inputs()

    def get_all_biases(self):
        return [layer.get_bias() for layer in self.layers]

    def set_all_functions(self, functions):
        if len(functions) != self.get_n_layers():
            raise ValueError('Wrong number of functions')
        for layer, function in zip(self.layers, functions):
            layer.set_function(function)

    def set_layer_function(self, layer_index, function):
        self.layers[layer_index].set_function(function) if layer_index < self.get_n_layers() else None

    def set_layer_weights(self, layer_index, weights):
        if not layer_index < self.get_n_layers():
            raise IndexError('Wrong layer index')

        if type(weights) is list:
            weights = np.array(weights)

        layer = self.layers[layer_index]
        if layer_index == 0:
            if self.get_n_inputs() != weights.shape[1]:
                raise ValueError('weights dimension is not coherent with inputs number')
            layer.set_weights(weights)
            return
        prev_layer = self.layers[layer_index - 1]
        if not prev_layer.are_next_weights_correct(weights):
            raise ValueError('Wrong weights dimension')
        layer.set_weights(weights)

    def set_neuron_weights(self, layer_index, neuron_index, weights):
        if type(weights) is list:
            weights = np.array(weights)
        if layer_index < self.get_n_layers():
            self.layers[layer_index].set_neuron_weights(neuron_index, weights)

    def set_neuron_weight(self, layer_index, neuron_index, ancestor_index, weight):
        if not layer_index < self.get_n_layers():
            raise IndexError('Wrong layer index')
        self.layers[layer_index].set_neuron_weight(neuron_index, ancestor_index, weight)

    def set_neurons_number(self, layer_index, n_neurons, weights, bias=0, next_weights=None):
        """enables changing number of neurons on the layer and setting weights"""
        if not layer_index < self.get_n_layers():
            raise IndexError('Wrong layer index')
        if n_neurons < 1:
            raise ValueError('Wrong number of neurons')
        if not (type(bias) is int or type(bias) is float):
            if len(bias) != n_neurons:
                raise ValueError('Wrong bias length')
        if next_weights is None and layer_index != self.get_n_layers() - 1:
            return AttributeError('Attribute next_weights is essential for this layer')
        self.layers[layer_index].set_n_neurons(n_neurons)
        self.set_layer_weights(layer_index, weights)
        self.layers[layer_index].set_bias(bias)
        if layer_index == self.get_n_layers() - 1:
            return
        self.set_layer_weights(layer_index + 1, next_weights)
