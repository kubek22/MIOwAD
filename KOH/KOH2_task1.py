#%% 

from pandas import read_csv
import matplotlib.pyplot as plt
import time
from model import SOM, HexagonalMesh
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

#%%

df = read_csv("data/mnist_train.csv")
df.head()

n_classes = len(df.label.value_counts())

data = df.iloc[:, 1:].to_numpy()
classes = df.label.to_numpy()

data = MinMaxScaler((0, 1)).fit_transform(data)

#%%

tsne = TSNE(n_components=2, random_state=10)
data_embedded = tsne.fit_transform(data)

#%%

# cmap = plt.colormaps['tab20']

tab20_colors = plt.cm.tab20.colors
tab20b_colors = plt.cm.tab20b.colors
combined_colors = tab20_colors + tab20b_colors

cmap = mcolors.ListedColormap(combined_colors)

plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(df["label"]))
plt.show()

#%% gauss, rect

som = SOM(5, 5, 784)

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
tsne = TSNE(n_components=2, random_state=10, perplexity=5)
weights_embedded = tsne.fit_transform(weights)
plt.scatter(weights_embedded[:, 0], weights_embedded[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% assigning classes

som.assign_classes(data, classes)
labels = som.labels
weights = som.weights
tsne = TSNE(n_components=2, random_state=10, perplexity=5)
weights_embedded = tsne.fit_transform(weights)
plt.scatter(weights_embedded[:, 0], weights_embedded[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% labeling vectors

c = som.predict(data)
plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(c))
plt.show()

#%% max silhouette score

silhouette_score(data, classes)

#%% hat, rect

som = SOM(5, 5, 784)
som.neighborhood_function = som.mexican_hat

start_time = time.time()

scores = som.fit(data, epochs=7, init_lr=1e-7, scale=0.2)

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

#%% basic neurons

labels = som.labels
weights = som.weights
tsne = TSNE(n_components=2, random_state=10, perplexity=5)
weights_embedded = tsne.fit_transform(weights)
plt.scatter(weights_embedded[:, 0], weights_embedded[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% assigning classes

som.assign_classes(data, classes)
labels = som.labels
weights = som.weights
tsne = TSNE(n_components=2, random_state=10, perplexity=5)
weights_embedded = tsne.fit_transform(weights)
plt.scatter(weights_embedded[:, 0], weights_embedded[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% labeling vectors

c = som.predict(data)
plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(c))
plt.show()

#%% gauss, hex

som = SOM(5, 5, 784)
som.mesh = HexagonalMesh(5, 5)

start_time = time.time()

scores = som.fit(data, epochs=10, init_lr=0.003, scale=0.8)

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

#%% basic neurons

labels = som.labels
weights = som.weights
tsne = TSNE(n_components=2, random_state=10, perplexity=5)
weights_embedded = tsne.fit_transform(weights)
plt.scatter(weights_embedded[:, 0], weights_embedded[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% assigning classes

som.assign_classes(data, classes)
labels = som.labels
weights = som.weights
tsne = TSNE(n_components=2, random_state=10, perplexity=5)
weights_embedded = tsne.fit_transform(weights)
plt.scatter(weights_embedded[:, 0], weights_embedded[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% labeling vectors

c = som.predict(data)
plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(c))
plt.show()

#%% hat, hex

som = SOM(5, 5, 784)
som.neighborhood_function = som.mexican_hat
som.mesh = HexagonalMesh(5, 5)

start_time = time.time()

scores = som.fit(data, epochs=10, init_lr=1e-10, scale=1) # (-9, 0.5) (-10, 1), (-11, 1)

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

#%% basic neurons

labels = som.labels
weights = som.weights
tsne = TSNE(n_components=2, random_state=10, perplexity=5)
weights_embedded = tsne.fit_transform(weights)
plt.scatter(weights_embedded[:, 0], weights_embedded[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% assigning classes

som.assign_classes(data, classes)
labels = som.labels
weights = som.weights
tsne = TSNE(n_components=2, random_state=10, perplexity=5)
weights_embedded = tsne.fit_transform(weights)
plt.scatter(weights_embedded[:, 0], weights_embedded[:, 1], color=cmap(labels))
plt.show()

som.get_silhouette_score(data)

#%% labeling vectors

c = som.predict(data)
plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(c))
plt.show()

