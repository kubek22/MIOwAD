#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math

#%% steps-large

df_train = read_csv("data/regression/steps-large-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/steps-large-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]


#%%
plt.plot(x_train, y_train, 'o')
plt.show()

#%%

plt.plot(x_test, y_test, 'o')
plt.show()

#%%
def sigma(x):
    return 1 / (1 + math.e ** ((-1) * x))

def MSE(x, y):
    return sum((x - y) ** 2) / len(x)

#%%

# Generate some example data
# x_train = np.linspace(-1.5, 2, 100).reshape(-1, 1)
# y_train = x_train ** 2

# Build a neural network with one layer and 5 neurons
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5, input_shape=(1,), activation='sigmoid'), # , use_bias=False
    tf.keras.layers.Dense(units=1)  # Output layer with linear activation
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=5000, verbose=0) # > 3000

# Retrieve the learned weights
# weights_layer1 = model.layers[0].get_weights()[0]
# weights_layer1 = model.layers[1].get_weights()[0]


# weights_output = model.layers[2].get_weights()[0]
# biases_output = model.layers[1].get_weights()[1]

# print("Weights and Biases for Layer 1:")
# print("Weights:")
# print(weights_layer1)
# print("Biases:")

# print("\nWeights and Biases for Output Layer:")
# print("Weights:")
# print(weights_output)
# print("Biases:")
# print(biases_output)

#%% train

pred = model.predict(x_train)
pred = np.reshape(pred, (1, 10000))[0]

plt.plot(x_train, y_train, 'o')
plt.plot(x_train, pred, 'o')
plt.show()

sum((pred - y_train) ** 2) / len(pred)

#%% test

pred = model.predict(x_test)
pred = np.reshape(pred, (1, 1000))[0]

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, pred, 'o')
plt.show()

sum((pred - y_test) ** 2) / len(pred)



