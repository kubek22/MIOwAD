#%% 

from pandas import read_csv
import matplotlib.pyplot as plt
import time
from model import SOM
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

#%%

df = read_csv("data/cube.csv")
df.head()

n_classes = len(df.c.value_counts())

data = df[["x", "y", "z"]].to_numpy()
classes = df.c.to_numpy()

data = MinMaxScaler((0, 1)).fit_transform(data)

tsne = TSNE(n_components=2, random_state=10, perplexity=5)
data_embedded = tsne.fit_transform(data)
data_embedded = MinMaxScaler((0, 1)).fit_transform(data_embedded)

#%%

cmap = plt.colormaps['tab20']
plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(df["c"]))
plt.show()

#%% 

som = SOM(2, 4, 3) # (2, 3) and (4, 5)

start_time = time.time()

scores = som.fit(data, epochs=20, init_lr=0.003, scale=0.8)

end_time = time.time()
time1 = end_time - start_time
print(time1)

#%%

plt.plot(scores)
plt.show()

#%% labeling vectors

c = som.predict(data)
plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(c))
plt.show()

#%% neurons

labels = som.labels
weights = som.weights
weights_embedded = tsne.fit_transform(weights)
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
plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(c))
plt.show()

#%% max silhouette score

silhouette_score(data, classes)

#%% mexican hat

som = SOM(2, 4, 3)
som.neighborhood_function = som.mexican_hat # requires smaller lr

start_time = time.time()

scores = som.fit(data, epochs=100, init_lr=0.0003, scale=0.9) # 0.8, 0.9, 1

end_time = time.time()
time1 = end_time - start_time
print(time1)

#%%

plt.plot(scores)
plt.show()

#%% labeling vectors

c = som.predict(data)
plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(c))
plt.show()

#%% neurons

labels = som.labels
weights = som.weights
weights_embedded = tsne.fit_transform(weights)
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
plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(c))
plt.show()

#%% max silhouette score

silhouette_score(data, classes)
