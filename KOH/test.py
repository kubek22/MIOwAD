import numpy as np
import time

k = int(1e5)
x = np.zeros((k, 2))
x = np.random.rand(k, 2)

start_time = time.time()
for r in x:
    r += 1
end_time = time.time()
time1 = end_time - start_time
print(time1)

start_time = time.time()
x - [1, 1]
end_time = time.time()
time2 = end_time - start_time
print(time2)

print(time1 / time2)

#%%

def f(x):
    return x ** 2

v_f = np.vectorize(f)

k = int(1e5)
x = np.random.rand(k)

start_time = time.time()
res = []
for el in x:
    res.append(f(el))
end_time = time.time()
time1 = end_time - start_time
print(time1)

start_time = time.time()
res = v_f(x)
end_time = time.time()
time2 = end_time - start_time
print(time2)
print(time1 / time2)

#%%

x = np.random.rand(10, 2)
w_x = x[np.random.randint(0, 10)]

np.sum((x - w_x) ** 2, axis=1)
np.argmin(np.sum((x - w_x) ** 2, axis=1))

#%%

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

np.random.shuffle(matrix)

print(matrix)  

#%%

array_1d = np.array([1, 2, 3, 4, 5])
array_2d = np.array([[2, 3],
                     [4, 5],
                     [6, 7],
                     [8, 9],
                     [10, 11]])

array_1d_reshaped = array_1d[:, np.newaxis]  # or array_1d[:, None]

result = array_1d_reshaped * array_2d

print(result)

