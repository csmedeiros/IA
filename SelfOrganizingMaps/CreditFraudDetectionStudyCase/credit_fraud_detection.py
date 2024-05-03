from minisom import MiniSom
import pandas as pd
import numpy as np

base = pd.read_csv("credit_data.csv")
base = base.dropna()
base.loc[base.age < 0, 'age'] = 40.92

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

som = MiniSom(x=15, y=15, input_len=4, random_seed=0, sigma=1, learning_rate=0.5)
som.train_random(data=X, num_iteration=10000)

from matplotlib.pylab import pcolor, colorbar, plot

pcolor(som.distance_map().T)

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize=10, markeredgecolor = colors[y[i]], markeredgewidth = 2)
    
mapping = som.win_map(X)
suspects = np.concatenate((mapping[(9, 11)], mapping[(13, 7)]), axis=0)
suspects = scaler.inverse_transform(suspects)

default = []
for i in range(len(base)):
    for j in range(len(suspects)):
        if base.iloc[i, 0] == suspects[j, 0]:
            default.append(base.iloc[i, 4])
            
default = np.asarray(default)

result = np.column_stack((suspects, default))
result = result[result[:, 4].argsort()]