from minisom import MiniSom
import pandas as pd
import numpy as np

X = pd.read_csv("entradas_breast.csv").values
y = pd.read_csv("saidas_breast.csv")
y = y.iloc[:, 0].values

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

som = MiniSom(x=11, y=11, input_len=30, sigma=1, learning_rate=0.5, random_seed=0)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100000)

from matplotlib.pylab import pcolor, colorbar, plot

pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']


for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize=10, markeredgecolor = colors[y[i]], markeredgewidth = 2)