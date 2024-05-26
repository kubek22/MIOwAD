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
    
    x = random.uniform(x_lb, x_rb)
    rect_x_mid = x + 0.5 * width
    
    y_ub = math.sqrt(r ** 2 - x ** 2)
    if rect_x_mid > 0:
        y_ub = math.sqrt(r ** 2 - (x + width) ** 2)
    y_lb = -y_ub + height
    y_ub = min(y_ub, y_up)
    y_lb = max(y_lb, y_down + height)
    if y_lb > y_ub:
        return None
    y = random.uniform(y_lb, y_ub)
    rect = [x, y, width, height, value]
    return rect

def get_rectangles_from_area(rectangles: np.array,
                     x_left: float, x_right: float, 
                     y_down: float, y_up: float) -> np.array:
    i = rectangles[:, 0] + rectangles[:, 2] > x_left
    rectangles = rectangles[i, :]
    i = rectangles[:, 0] < x_right
    rectangles = rectangles[i, :]
    i = rectangles[:, 1] - rectangles[:, 3] > y_down
    rectangles = rectangles[i, :]
    i = rectangles[:, 1] < y_up
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
        direction = np.random.randint(4)
        if direction == LEFT:
            x_right = x
        elif direction == RIGHT:
            x_left = x + width
        elif direction == DOWN:
            y_up = y - height
        elif direction == UP:
            y_down = y
        rectangles_copy = get_rectangles_from_area(rectangles_copy, x_left, x_right, y_down, y_up)
    new_rectangle = insert_rectangle(rectangle, r, x_left, x_right, y_down, y_up)
    if new_rectangle is None:
        return rectangles
    if len(rectangles) == 0:
        return np.array([new_rectangle])
    return np.append(rectangles, new_rectangle)

def initialize_population(size: int, available_rectangles: pd.DataFrame, r: float) -> List[np.array]:
    population = []
    n = len(rectangles)
    for i in range(size):
        rectangle = rectangles.iloc[np.random.randint(n), :]
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
            y_max = np.min(rectangles_above_y)
        
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
            x_min = np.max(rectangles_left_x)
        
        rectangles[i, 0] = x_min
        
    return rectangles

def mutation():
    pass

def crossbreeding():
    pass

def evaluation():
    pass

def selection():
    pass

def epoch():
    pass

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
    

