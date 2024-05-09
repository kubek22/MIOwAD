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

df = read_csv("data/UCI HAR Dataset/train/X_train.txt", sep='\s+', header=None)
df.head()

c = read_csv("data/UCI HAR Dataset/train/y_train.txt", sep='\s+', header=None)

n_classes = len(c.value_counts())

data = df.to_numpy()
classes = c[0].to_numpy()
classes -= 1

data = MinMaxScaler((0, 1)).fit_transform(data)

#%%

tsne = TSNE(n_components=2, random_state=10)
data_embedded = tsne.fit_transform(data)

#%%

tab20_colors = plt.cm.tab20.colors
tab20b_colors = plt.cm.tab20b.colors
combined_colors = tab20_colors + tab20b_colors

cmap = mcolors.ListedColormap(combined_colors)

plt.scatter(data_embedded[:, 0], data_embedded[:, 1], color=cmap(classes))
plt.show()

#%% gauss, rect

som = SOM(4, 4, 561)

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

som = SOM(4, 4, 561)
som.neighborhood_function = som.mexican_hat

start_time = time.time()

scores = som.fit(data, epochs=10, init_lr=1e-3, scale=0.5) # (1e-3, 0.5), (1e-6, 0.6), (1e-9, 0.8)

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

som = SOM(4, 4, 561)
som.mesh = HexagonalMesh(4, 4)

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

som = SOM(4, 4, 561)
som.neighborhood_function = som.mexican_hat
som.mesh = HexagonalMesh(4, 4)

start_time = time.time()

scores = som.fit(data, epochs=10, init_lr=1e-7, scale=4) # (4, 1e-6, 0.5), (1e-7, 0.4), (1e-9, 0.8) 

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

