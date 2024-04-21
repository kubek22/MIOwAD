#%% 

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import math
import pickle
import time
from sklearn.model_selection import train_test_split
from model import SOM

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

df_train, df_test = train_test_split(df, train_size=0.75, random_state=48, stratify=df.c)

train_data = df_train[["x", "y"]].to_numpy()
test_data = df_test[["x", "y"]].to_numpy()

#%%

cmap = plt.cm.get_cmap('tab10', 6)
plt.scatter(df_train["x"], df_train["y"], color=cmap(df_train["c"]))
plt.show()

#%%

som = SOM(4, 5, 2)

start_time = time.time()

som.fit(train_data, epochs=1000)

end_time = time.time()
time1 = end_time - start_time
print(time1)
