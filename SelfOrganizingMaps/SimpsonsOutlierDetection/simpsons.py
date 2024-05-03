import pandas as pd
from minisom import MiniSom
import numpy as np

base = pd.read_csv("personagens.csv")
base = base.dropna()

X = base.iloc[:, 0:6].values
y = base.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


som = MiniSom(x=9, y=9, input_len=6, random_seed=0)
som.random_weights_init(X)
som.train_random(X, num_iteration=1000)

from matplotlib.pylab import pcolor, colorbar, plot

pcolor(som.distance_map())

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize=10, markeredgecolor = colors[y[i]], markeredgewidth = 2)
    
mapping = som.win_map(X)
suspects = np.concatenate((mapping[(5, 8)]), axis=0)
suspects = scaler.inverse_transform(suspects)