#%% 

import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.colors as mcolors
import os
import re
import random
from typing import Dict, List, Tuple
import math
import copy

#%%

def read_rectangles(path: str, file_name: str) -> Tuple[int, pd.DataFrame]:
    radius = int(re.search(r'\d+', file_name)[0])
    file_path = os.path.join(path, file_name)
    rectangles = read_csv(file_path, header=None)
    rectangles.head()
    rectangles.columns = ['width', 'height', 'value']
    return radius, rectangles

def plot_circle(r: float, a: float = 0, b: float = 0):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)
    margin = 0.1 * r
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y)
    ax.set_xlim(a - r - margin, a + r + margin)
    ax.set_ylim(b - r - margin, b + r + margin)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    return fig, ax

def plot_rectangle(ax, height: float, width: float, x: float = 0, y: float = 0):
    y_lower_left = y - height
    rect = plt.Rectangle((x, y_lower_left), width, height, fill=False, edgecolor='r')
    ax.add_patch(rect)

def plot_individual(radius: int, individual: np.array):
    fig, ax = plot_circle(radius)
    for rectangle in individual:
        x = rectangle[0]
        y = rectangle[1]
        width = rectangle[2]
        height = rectangle[3]
        plot_rectangle(ax, height, width, x, y)
    
def intersect_intervals(x1: float, x2: float, y1: float, y2: float) -> Tuple[float, float]:
    lb = max(x1, y1)
    rb = min(x2, y2)
    return (lb, rb)

def sum_intervals(x1: float, x2: float, y1: float, y2: float) -> Tuple[float, float]:
    """
    assumes that the sum is one interval
    """
    if x1 > x2:
        return (y1, y2)
    if y1 > y2:
        return (x1, x2)
    intersection = intersect_intervals(x1, x2, y1, y2)
    if intersection[0] > intersection[1]:
        return (math.inf, -math.inf)
    lb = min(x1, y1)
    rb = max(x2, y2)
    return (lb, rb)

def solve_quadratic_inequality(c: float) -> Tuple[float, float]:
    """
    returns solution interval of inequality:
        x ** 2 + c ** 2 <= 0
    """
    lb = math.inf
    rb = -math.inf
    delta = -4 * c
    if delta < 0:
        return (lb, rb)
    sqrt_delta = math.sqrt(delta)
    lb = -sqrt_delta / 2
    rb = -lb
    return (lb, rb)
    
def adapt_x_interval_to_y_constraints(x_lb: float, x_rb: float, y_down: float, y_up: float, 
                                      r: float, width: float, height: float) -> Tuple[float, float]:
    domain_interval_13 = intersect_intervals(x_lb, x_rb, -math.inf, -0.5 * width)
    domain_interval_24 = intersect_intervals(x_lb, x_rb, -0.5 * width, math.inf)
    solution_lb = x_lb
    solution_rb = x_rb
    solution_interval = (solution_lb, solution_rb)
    
    ## 1
    if height + y_down < 0:
        lb = -math.inf
        rb = math.inf
    else:
        c = -r ** 2 + (height + y_down) ** 2
        lb, rb = solve_quadratic_inequality(c)
    interval_1 = intersect_intervals(*domain_interval_13, lb, rb)
    
    ## 2
    if height + y_down < 0:
        lb = -math.inf
        rb = math.inf
    else:
        c = -r ** 2 + (height + y_down) ** 2
        lb, rb = solve_quadratic_inequality(c)
        lb -= width
        rb -= width
    interval_2 = intersect_intervals(*domain_interval_24, lb, rb)
    interval_12 = sum_intervals(*interval_1, *interval_2)
    
    ## 3
    if height - y_up < 0:
        lb = -math.inf
        rb = math.inf
    else:
        c = -r ** 2 + (height - y_up) ** 2
        lb, rb = solve_quadratic_inequality(c)
    interval_3 = intersect_intervals(*domain_interval_13, lb, rb)
    
    ## 4
    if height - y_up < 0:
        lb = -math.inf
        rb = math.inf
    else:
        c = -r ** 2 + (height - y_up) ** 2
        lb, rb = solve_quadratic_inequality(c)
        lb -= width
        rb -= width
    interval_4 = intersect_intervals(*domain_interval_24, lb, rb)
    interval_34 = sum_intervals(*interval_3, *interval_4)
    
    solution_interval = intersect_intervals(*interval_12, *interval_34)
    
    return solution_interval

def insert_rectangle(rectangle: pd.Series, r: float, 
                     x_left: float = -math.inf, x_right: float = math.inf, 
                     y_down: float = -math.inf, y_up: float = math.inf) -> np.array:
    width = rectangle[0]
    height = rectangle[1]
    value = rectangle[2]
    x_lb = -1 * math.sqrt(r ** 2 - (height / 2) ** 2)
    x_rb = -x_lb - width
    x_lb = max(x_lb, x_left)
    x_rb = min(x_rb, x_right - width)
    if x_lb > x_rb:
        return None
    
    x_lb, x_rb = adapt_x_interval_to_y_constraints(x_lb, x_rb, y_down, y_up, r, width, height)
    if x_lb > x_rb:
        return None
    
    x = np.random.uniform(x_lb, x_rb)
    rect_x_mid = x + 0.5 * width
    
    y_ub = math.sqrt(r ** 2 - x ** 2)
    if rect_x_mid > 0:
        y_ub = math.sqrt(r ** 2 - (x + width) ** 2)
    y_lb = -y_ub + height
    y_ub = min(y_ub, y_up)
    y_lb = max(y_lb, y_down + height)
    if y_lb > y_ub:
        return None
    y = np.random.uniform(y_lb, y_ub)
    rect = [x, y, width, height, value]
    return rect

def get_rectangles_from_area(rectangles: np.array,
                     x_left: float = -math.inf, x_right: float = math.inf, 
                     y_down: float = -math.inf, y_up: float = math.inf) -> np.array:
    i = rectangles[:, 0] + rectangles[:, 2] > x_left
    rectangles = rectangles[i, :]
    i = rectangles[:, 0] < x_right
    rectangles = rectangles[i, :]
    i = rectangles[:, 1] > y_down
    rectangles = rectangles[i, :]
    i = rectangles[:, 1] - rectangles[:, 3] < y_up
    rectangles = rectangles[i, :]
    return rectangles

def randomly_insert_rectangle(rectangle: pd.Series, rectangles: np.array, r: float, 
                     x_left: float = -math.inf, x_right: float = math.inf, 
                     y_down: float = -math.inf, y_up: float = math.inf):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    rectangles_copy = copy.copy(rectangles)
    while len(rectangles_copy) != 0:
        rect = rectangles_copy[0, :]
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        
        x_left_copy = x_left
        x_right_copy = x_right
        y_down_copy = y_down
        y_up_copy = y_up
        
        available_directions = [LEFT, RIGHT, UP, DOWN]
        probabilities = [0.05, 0.45, 0.05, 0.45]
        test_rectangle = None  # checks if chosen space is big enough to fit the rectangle
        while len(available_directions) > 0 and test_rectangle is None:
            x_left = x_left_copy
            x_right = x_right_copy
            y_down = y_down_copy
            y_up = y_up_copy
            
            direction = random.choices(available_directions, weights=probabilities)[0]
            d_index = np.where(np.array(available_directions) == direction)[0][0]
            available_directions.pop(d_index)
            probabilities.pop(d_index)
            if direction == LEFT:
                x_right = x
            elif direction == RIGHT:
                x_left = x + width
            elif direction == DOWN:
                y_up = y - height
            elif direction == UP:
                y_down = y
            test_rectangle = insert_rectangle(rectangle, r, x_left, x_right, y_down, y_up)
        rectangles_copy = get_rectangles_from_area(rectangles_copy, x_left, x_right, y_down, y_up)
        
    new_rectangle = insert_rectangle(rectangle, r, x_left, x_right, y_down, y_up)
    if new_rectangle is None:
        return rectangles
    if len(rectangles) == 0:
        return np.array([new_rectangle])
    return np.append(rectangles, np.array([new_rectangle]), axis=0)

def initialize_population(size: int, available_rectangles: pd.DataFrame, r: float) -> List[np.array]:
    population = []
    n = len(available_rectangles)
    for i in range(size):
        rectangle = available_rectangles.iloc[np.random.randint(n), :]
        individual = randomly_insert_rectangle(rectangle, np.array([]), r)
        population.append(individual)
    return population

def shift_rectangles(rectangles: np.array, r: float) -> np.array:
    # shift up
    indices = np.argsort(rectangles[:, 1])[::-1]
    rectangles = rectangles[indices]
    for i in range(len(rectangles)):
        rectangle = rectangles[i, :]
        x = rectangle[0]
        y = rectangle[1]
        width = rectangle[2]
        height = rectangle[3]
        
        rect_x_mid = x + 0.5 * width
        y_max = math.sqrt(r ** 2 - x ** 2)
        if rect_x_mid > 0:
            y_max = math.sqrt(r ** 2 - (x + width) ** 2)
        
        rectangles_above = get_rectangles_from_area(rectangles, x, x + width, y, math.inf)
        if len(rectangles_above) > 0:
            rectangles_above_y = rectangles_above[:, 1] - rectangles_above[:, 3]
            y_max = min(np.min(rectangles_above_y), y_max)
        
        rectangles[i, 1] = y_max
    
    # shift left
    indices = np.argsort(rectangles[:, 0])[::-1]
    rectangles = rectangles[indices]
    for i in range(len(rectangles)):
        rectangle = rectangles[i, :]
        x = rectangle[0]
        y = rectangle[1]
        width = rectangle[2]
        height = rectangle[3]
        
        rect_y_mid = y - 0.5 * height
        x_min = -math.sqrt(r ** 2 - y ** 2)
        if rect_y_mid < 0:
            x_min = -math.sqrt(r ** 2 - (y - height) ** 2)
        
        rectangles_left = get_rectangles_from_area(rectangles, -math.inf, x, y - height, y)
        if len(rectangles_left) > 0:
            rectangles_left_x = rectangles_left[:, 0] + rectangles_left[:, 2]
            x_min = max(np.max(rectangles_left_x), x_min)
        
        rectangles[i, 0] = x_min
        
    return rectangles

def mutation(population: List[np.array], available_rectangles: pd.DataFrame, r: float) -> List[np.array]:
    n = len(available_rectangles)
    for i in range(len(population)):
        individual = population[i]
        individual = shift_rectangles(individual, r)
        
        rectangle = available_rectangles.iloc[np.random.randint(n), :]
        individual = randomly_insert_rectangle(rectangle, individual, r)
        population[i] = individual
    return population

def get_full_rectangles_from_area(rectangles: np.array,
                     x_left: float = -math.inf, x_right: float = math.inf, 
                     y_down: float = -math.inf, y_up: float = math.inf) -> np.array:
    i = rectangles[:, 0] >= x_left
    rectangles = rectangles[i, :]
    i = rectangles[:, 0] + rectangles[:, 2] <= x_right
    rectangles = rectangles[i, :]
    i = rectangles[:, 1] - rectangles[:, 3] >= y_down
    rectangles = rectangles[i, :]
    i = rectangles[:, 1] <= y_up
    rectangles = rectangles[i, :]
    return rectangles

def merge_individuals(individual_part1: np.array, individual_part2: np.array) -> np.array:
    if len(individual_part1) == 0:
        new_individual = individual_part2
    elif len(individual_part2) == 0:
        new_individual = individual_part1
    else:
        new_individual = np.append(individual_part1, individual_part2, axis=0)
    return new_individual

def crossbreeding(population: List[np.array], r: float, proba: float = 1) -> List[np.array]:
    # TODO check
    n = len(population)
    for i in range(n - 1, -1, -1):
        if np.random.uniform() > proba:
            continue
        j = np.random.choice(n)
        use_horizontal_line = bool(np.random.randint(2))
        if use_horizontal_line:
            y = np.random.uniform(-r, r)
            individual_i_down = get_full_rectangles_from_area(population[i], y_up=y)
            individual_i_up = get_full_rectangles_from_area(population[i], y_down=y)
            individual_j_down = get_full_rectangles_from_area(population[j], y_up=y)
            individual_j_up = get_full_rectangles_from_area(population[j], y_down=y)
            
            new_individual_i = merge_individuals(individual_i_down, individual_j_up)
            new_individual_j = merge_individuals(individual_j_down, individual_i_up)
        else:
            x = np.random.uniform(-r, r)
            individual_i_left = get_full_rectangles_from_area(population[i], x_right=x)
            individual_i_right = get_full_rectangles_from_area(population[i], x_left=x)
            individual_j_left = get_full_rectangles_from_area(population[j], x_right=x)
            individual_j_right = get_full_rectangles_from_area(population[j], x_left=x)
            
            new_individual_i = merge_individuals(individual_i_left, individual_j_right)
            new_individual_j = merge_individuals(individual_j_left, individual_i_right)
        
        population[i] = new_individual_i
        population[j] = new_individual_j
    return population

def evaluation(population: List[np.array]):
    results = []
    for individual in population:
        results.append(np.sum(individual[:, 4]))
    return results

def selection(population: List[np.array], eval_results: List[float], size: int, best_fraction: float):
    best_number = min(int(size * best_fraction), size)
    random_number = size - best_number
    order = np.argsort(eval_results)[::-1]
    
    population = [population[i] for i in order]
    new_population = [population[order[i]] for i in range(best_number)]
    
    eval_results = np.array(eval_results) / np.sum(eval_results)
    random_indices = np.random.choice(size, random_number, p=eval_results)
    
    for i in random_indices:
        new_population.append(population[i])
    
    return new_population

def epoch():
    radius, available_rectangles = read_rectangles("data/cutting", "r800.csv")
    size = 3
    population = initialize_population(size, available_rectangles, radius)
    for individual in population:
        plot_individual(radius, individual)
    
    population = mutation(population, available_rectangles, radius)
    for individual in population:
        plot_individual(radius, individual)
        
    population = crossbreeding(population, radius)
    for individual in population:
        plot_individual(radius, individual)
    
    eval_results = evaluation(population)
    population = selection(population, eval_results, size, 0)
    for individual in population:
        plot_individual(radius, individual)
    

#%%

radius, rectangles = read_rectangles("data/cutting", "r800.csv")

# individual = np.random.rand(10, 4)
# plot_individual(1, individual)

# radius = 200
# individual = randomly_insert_rectangle(rectangles.loc[1,:], np.array([]), radius, y_up=60, y_down=-180, x_left=-150, x_right=170)
# plot_individual(radius, individual)

population = initialize_population(3, rectangles, radius)
for individual in population:
    plot_individual(radius, individual)
    

