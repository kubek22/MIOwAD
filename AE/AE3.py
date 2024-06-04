#%%

import sys
project_dir = '..'
sys.path.append(project_dir)

from NN.model import Net
from ucimlrepo import fetch_ucirepo
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time

#%%

def predict_class(predictions):
    classes = []
    for p in predictions:
        # classes.append(np.where(np.max(p) == p)[0][0])
        classes.append(np.argmax(p))
    return np.array(classes).reshape(-1)

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

def initialize_population(size, n_neurons, n_inputs, functions, use_softmax):
    population = []
    for i in range(size):
        net =  Net(n_neurons=n_neurons, n_inputs=n_inputs, functions=functions, param_init='xavier', use_softmax=use_softmax)
        population.append(net)
    return population

def mutation(population, mean=0, std_dev=1):
    for i in range(len(population)):
        net = population[i]
        functions = net.get_all_functions()
        weights = net.get_all_weights()
        biases = net.get_all_biases()
        new_weights = []
        for layer_weights in weights:
            n = layer_weights.shape[0] * layer_weights.shape[1]
            noise = np.random.normal(mean, std_dev, n)
            noise = noise.reshape(layer_weights.shape[0], layer_weights.shape[1])
            new_weights.append(layer_weights + noise)
        new_biases = []
        for layer_biases in biases:
            n = len(layer_biases)
            noise = np.random.normal(mean, std_dev, n)
            new_biases.append(layer_biases + noise)
        net.set_all_weights(new_weights)
        net.set_all_biases(new_biases)
        # use_softmax = net.use_softmax
        # net = Net(new_weights, functions, new_biases, use_softmax=use_softmax)
        # population[i] = net
    return population

def crossbreeding(population):
    size = len(population)
    n_layers = population[0].get_n_layers()
    for i in range(size):
        j = np.random.choice(size)
        net_i = population[i]
        net_j = population[j]
        functions = net_i.get_all_functions()
        # weights
        for layer_idx in range(n_layers):
            w_i = net_i.get_layer_weights(layer_idx)
            w_j = net_j.get_layer_weights(layer_idx)
            matrix_size = (w_i.shape[0], w_i.shape[1])
            replace_idx = np.arange(w_i.shape[0] * w_i.shape[1])
            num_elements_to_select = np.random.choice(replace_idx)
            
            random_indices = np.random.choice(replace_idx, num_elements_to_select, replace=False)

            random_indices = np.unravel_index(random_indices, matrix_size)
            random_indices_pairs = list(zip(random_indices[0], random_indices[1]))
            
            for indices_pairs in random_indices_pairs:
                temp = w_i[indices_pairs]
                w_i[indices_pairs] = w_j[indices_pairs]
                w_j[indices_pairs] = temp
                
            net_i.set_layer_weights(layer_idx, w_i)
            net_j.set_layer_weights(layer_idx, w_j)
        # biases
        biases_i = net_i.get_all_biases()
        biases_j = net_j.get_all_biases()
        new_biases_i = []
        new_biases_j = []
        for layer_idx in range(n_layers):
            b_i = biases_i[layer_idx]
            b_j = biases_j[layer_idx]
            replace_idx = np.arange(len(b_i))
            num_elements_to_select = np.random.choice(replace_idx)
            replace_idx = np.random.choice(replace_idx, num_elements_to_select)
            temp = b_i[replace_idx]
            b_i[replace_idx] = b_j[replace_idx]
            b_j[replace_idx] = temp
            new_biases_i.append(b_i)
            new_biases_j.append(b_j)
            
        net_i.set_all_biases(new_biases_i)
        net_j.set_all_biases(new_biases_j)
        
        # weights_i = net_i.get_all_weights()
        # weights_j = net_j.get_all_weights()
        # use_softmax = net_i.use_softmax
        # net_i = Net(weights_i, functions, biases_i, use_softmax=use_softmax)
        # use_softmax = net_j.use_softmax
        # net_j = Net(weights_j, functions, biases_j, use_softmax=use_softmax)
        # population[i] = net_i
        # population[j] = net_j
    return population

def evaluation(population, x_train, y_train, x_test, y_test, classification):
    eval_results = []
    eval_test_results = []
    for net in population:
        predictions = []
        for x in x_train:
            predictions.append(net.predict(x))
        if classification:
            predictions = predict_class(predictions)
        mse = MSE(predictions, y_train)
        eval_results.append(mse)
        
        test_predictions = []
        for x in x_test:
            test_predictions.append(net.predict(x))
        if classification:
            test_predictions = predict_class(test_predictions)
        mse = MSE(test_predictions, y_test)
        eval_test_results.append(mse)
    return eval_results, eval_test_results

def selection(population, eval_res, best_fraction):
    order = np.argsort(eval_res)
    size = len(population)
    best_number = min(int(size * best_fraction), size)
    random_number = size - best_number
    sorted_population = [population[order[i]] for i in range(len(order))]
    new_population = sorted_population[:best_number]
    p = eval_res / np.sum(eval_res)
    random_indices = np.random.choice(size, random_number, p=p)
    for i in random_indices:
        new_population.append(population[i])
    return new_population

def epoch(population, best_fraction, x_train, y_train, x_test, y_test, classification):
    population = mutation(population)
    population = crossbreeding(population)
    eval_res, eval_test_res = evaluation(population, x_train, y_train, x_test, y_test, classification)
    new_population = selection(population, eval_res, best_fraction)
    return new_population, eval_res, eval_test_res

def run(epochs, size, n_neurons, n_inputs, functions, best_fraction, x_train, y_train, x_test, y_test, use_softmax=False):
    classification = use_softmax
    population = initialize_population(size, n_neurons, n_inputs, functions, use_softmax)
    best_results = []
    best_test_results = []
    start_time = time.time()
    for e in range(epochs):
        population, eval_res, eval_test_res = epoch(population, best_fraction, x_train, y_train, x_test, y_test, classification)
        best_results.append(min(eval_res))
        best_test_results.append(min(eval_test_res))
        print(e)
        print(time.time() - start_time)
    final_results, final_test_results = evaluation(population, x_train, y_train, x_test, y_test, classification)
    best_idx = np.argmin(final_results)
    best_net = population[best_idx]
    
    best_test_idx = np.argmin(final_test_results)
    best_test_net = population[best_idx]
    
    plt.plot(best_results)
    plt.plot(best_test_results)
    plt.show()
    
#%% iris

iris = fetch_ucirepo(id=53) 
  
x = iris.data.features.to_numpy()
y = iris.data.targets
y = y.to_numpy().reshape(-1)
  
le = LabelEncoder()
le.fit(y)
le.classes_
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

run(epochs=0, size=10, n_neurons=[5, 5, 3], n_inputs=4, functions=['tanh', 'tanh', 'softmax'], best_fraction=0.2, 
    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, use_softmax=True)

#%% multimodal-large

df_train = read_csv("data/multimodal-large-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/multimodal-large-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

run(epochs=0, size=10, n_neurons=[5, 5, 1], n_inputs=1, functions=['tanh', 'tanh', 'linear'], best_fraction=0.2, 
    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

#%%

auto_mpg = fetch_ucirepo(id=9) 
  
x = auto_mpg.data.features.to_numpy()
y = auto_mpg.data.targets 
y = y.to_numpy().reshape(-1) 

idx = ~np.isnan(x).any(axis=1)

x = x[idx]
y = y[idx]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

run(epochs=3, size=5, n_neurons=[5, 5, 1], n_inputs=7, functions=['tanh', 'tanh', 'linear'], best_fraction=0.2, 
    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, use_softmax=False)

