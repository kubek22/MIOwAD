import math
from autograd import grad
import numpy as np

def f(x):
    return x**2 + 3*x + 2

df_dx = grad(f)

x_value = 2.0
derivative_value = df_dx(x_value)
print(derivative_value)


#%%

from autograd import grad

def f(x, y, z):
    return x**2 + 3*y + 2*z

df_dx = grad(f, 0)
df_dy = grad(f, 1)
df_dz = grad(f, 2)

#%%

def sigma(x):
    if x > 0:
        return 1 / (1 + math.e ** ((-1) * x))
    return math.e ** x / (1 + math.e ** x)

#%%

df_dx = grad(sigma)
df_dx(0.0)
df_dx(1.0)
df_dx(-1.0)

#%%

f = np.vectorize(grad(sigma))

f(0.0)

