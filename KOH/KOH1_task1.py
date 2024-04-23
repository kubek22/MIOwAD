#%% 

from pandas import read_csv
import matplotlib.pyplot as plt
import pickle
import time
# from sklearn.model_selection import train_test_split
from model import SOM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

#%%

def save(array, file_name):
    file= open(file_name, 'wb')
    pickle.dump(array, file)
    file.close()

def read(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

#%%

df = read_csv("data/hexagon.csv")
df.head()

n_classes = len(df.c.value_counts())

# df_train, df_test = train_test_split(df, train_size=0.75, random_state=48, stratify=df.c)

# train_data = df_train[["x", "y"]].to_numpy()
# test_data = df_test[["x", "y"]].to_numpy()

data = df[["x", "y"]].to_numpy()
classes = df.c.to_numpy()

data = MinMaxScaler((0, 1)).fit_transform(data)

#%%

cmap = plt.colormaps['tab20']
plt.scatter(df["x"], df["y"], color=cmap(df["c"]))
plt.show()

#%% 

som = SOM(2, 3, 2)

start_time = time.time()

scores = som.fit(data, epochs=10, init_lr=0.003, scale=1)

end_time = time.time()
time1 = end_time - start_time
print(time1)

#%% mexican hat

som = SOM(2, 3, 2)
som.neighborhood_function = som.mexican_hat # requires smaller lr

start_time = time.time()

scores = som.fit(data, epochs=100, init_lr=0.0003, scale=0.5) # 0.1, 0.8, but not only

end_time = time.time()
time1 = end_time - start_time
print(time1)

#%%

plt.plot(scores)
plt.show()

#%% assigning classes

c = som.predict(data)
plt.scatter(data[:, 0], data[:, 1], color=cmap(c))
plt.show()

#%% neurons

labels = som.labels
weights = som.weights
plt.scatter(weights[:, 0], weights[:, 1], color=cmap(labels))
plt.show()

#%% silhouette score

som.get_silhouette_score(data)
silhouette_score(data, classes)
