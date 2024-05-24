import numpy as np

def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])

def initialize_population(size: int, dims: int):
    population = np.random.randn(size, dims)
    return population

def crossbreeding(population: np.array):
    size = population.shape[0]
    dims = population.shape[1]
    for i in range(len(population)):
        j = np.random.choice(size)
        replace_idx = np.arange(dims)
        num_elements_to_select = np.random.choice(replace_idx)
        replace_idx = np.random.choice(replace_idx, num_elements_to_select)
        temp = population[i][replace_idx]
        population[i][replace_idx] = population[j][replace_idx]
        population[j][replace_idx] = temp
    return population

def mutation(population: np.array, mean: float, std_dev: float):
    n = population.shape[0] * population.shape[1]
    noise = np.random.normal(mean, std_dev, n)
    noise = noise.reshape(population.shape[0], population.shape[1])
    return population + noise

def evaluation(population: np.array, function):
    res = np.array([])
    for subject in population:
        res = np.append(res, function(subject))
    return res

def selection(population: np.array, eval_res: np.array, size: int, best_fraction: float):
    order = np.argsort(eval_res)
    size = population.shape[0]
    dims = population.shape[1]
    best_number = min(int(size * best_fraction), size)
    random_number = size - best_number
    new_population = np.zeros((size, dims))
    new_population[:best_number] = population[order][:best_number]
    random_indices = np.random.choice(size, random_number)
    new_population[best_number:] = population[random_indices]
    return new_population

def epoch(size, dims, noise_mean, noise_std_dev, function, best_fraction, population=None):
    if population is None:
        population = initialize_population(size, dims)
    population = crossbreeding(population)
    population = mutation(population, noise_mean, noise_std_dev)
    eval_res = evaluation(population, function)
    new_population = selection(population, eval_res, size, best_fraction)
    return new_population, eval_res

def main():
    POPULATION_SIZE = 100
    DIMS = 3
    NOISE_MEAN = 0
    NOISE_STD_DEV = 1
    # FUNCTION = lambda a: a[0] ** 2 + a[1] ** 2 + 2 * a[2] ** 2
    FUNCTION = rastrigin
    BEST_FRACTION = 0.3
    EPOCHS = 100
    population = None
    best_subject = None
    best_result = np.inf
    for i in range(EPOCHS):
        population, results = epoch(POPULATION_SIZE, DIMS, NOISE_MEAN, NOISE_STD_DEV, FUNCTION, BEST_FRACTION, population)
        arg_min = np.argmin(results)
        if results[arg_min] < best_result:
            best_result = results[arg_min]
            best_subject = population[arg_min]
    # print('Final population: \n{}'.format(population))
    # print('Final results: \n{}'.format(results))
    print('Best subject: \n{}'.format(population[arg_min]))
    print('Best result: \n{}'.format(results[arg_min]))
    print('Global best subject: \n{}'.format(best_subject))
    print('Global best result: \n{}'.format(best_result))


if __name__ == '__main__':
    main()
