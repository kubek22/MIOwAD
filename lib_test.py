#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

#%%
df_train = read_csv("data/regression/square-simple-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]

df_test = read_csv("data/regression/square-simple-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]

#%%

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(1,)),
    tf.keras.layers.Dense(units=5, activation='sigmoid'), # , use_bias=False
    tf.keras.layers.Dense(units=1)  # Output layer with linear activation
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

#%% train

# Train the model
model.fit(x_train, y_train, epochs=1000, verbose=0, use_multiprocessing=True)

pred = model.predict(x_train)

plt.plot(x_train, y_train, 'o')
plt.plot(x_train, pred, 'o')
plt.show()

sum((np.transpose(pred)[0] - y_train) ** 2) / len(pred)

#%%

# Generate some example data
x_train = np.linspace(-1.5, 2, 100).reshape(-1, 1)
y_train = x_train ** 2

# Build a neural network with one layer and 5 neurons
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5, input_shape=(1,), activation='sigmoid'), # , use_bias=False
    tf.keras.layers.Dense(units=1)  # Output layer with linear activation
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=3700, verbose=0)

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

#pred = pred * 6 - 1.7

plt.plot(x_train, y_train, 'o')
plt.plot(x_train, pred, 'o')
plt.show()

sum((pred - y_train) ** 2) / len(pred)


#%%
df_train = read_csv("data/regression/square-simple-training.csv")
df_train.head()

x_train = df_train["x"]
y_train = df_train["y"]
y = x_train ** 2

df_test = read_csv("data/regression/square-simple-test.csv")
df_test.head()

x_test = df_test["x"]
y_test = df_test["y"]



#%%

x_test
y_test

pred = model.predict(x_test)
pred = np.reshape(pred, (1, 100))[0]
pred = pred * 90 - 130

sum((pred - y_test) ** 2) / len(pred)


#%% test
pred = model.predict(x_test)
pred = np.reshape(pred, (1, 100))[0]

#pred = pred * 6 - 1.7

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, pred, 'o')
plt.show()

sum((pred - y_test) ** 2) / len(pred)
