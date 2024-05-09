#%% 

from pandas import read_csv
import matplotlib.pyplot as plt
import time
from model import SOM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

#%%

df = read_csv("data/hexagon.csv")
df.head()

n_classes = len(df.c.value_counts())

data = df[["x", "y"]].to_numpy()
classes = df.c.to_numpy()

data = MinMaxScaler((0, 1)).fit_transform(data)

#%%

cmap = plt.colormaps['tab20']
plt.scatter(data[:, 0], data[:, 1], color=cmap(df["c"]))
plt.show()

#%% gauss

som = SOM(2, 3, 2) # (2, 3) and (4, 5)

start_time = time.time()

scores = som.fit(data, epochs=10, init_lr=0.003, scale=1)

end_time = time.time()
time1 = end_time - start_time
print(time1)

#%%

plt.plot(scores)
plt.show()

#%% labeling vectors

c = som.predict(data)
plt.scatter(data[:, 0], data[:, 1], color=cmap(c))
plt.show()

#%% basic neurons

labels = som.labels
weights = som.weights
plt.scatter(weights[:, 0], weights[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% assigning classes

som.assign_classes(data, classes)
labels = som.labels
weights = som.weights
plt.scatter(weights[:, 0], weights[:, 1], color=cmap(labels))
plt.show()

#%% labeling vectors

c = som.predict(data)
plt.scatter(data[:, 0], data[:, 1], color=cmap(c))
plt.show()

som.get_silhouette_score(data)

#%% max silhouette score

silhouette_score(data, classes)

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

#%% labeling vectors

c = som.predict(data)
plt.scatter(data[:, 0], data[:, 1], color=cmap(c))
plt.show()

#%% basic neurons

labels = som.labels
weights = som.weights
plt.scatter(weights[:, 0], weights[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% assigning classes

som.assign_classes(data, classes)
labels = som.labels
weights = som.weights
plt.scatter(weights[:, 0], weights[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% labeling vectors

c = som.predict(data)
plt.scatter(data[:, 0], data[:, 1], color=cmap(c))
plt.show()

#%% max silhouette score

silhouette_score(data, classes)
