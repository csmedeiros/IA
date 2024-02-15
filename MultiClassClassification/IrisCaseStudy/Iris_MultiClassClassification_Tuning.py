import pandas as pd
import numpy as np
import tensorflow.python as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import tensorflow.python.keras
from tensorflow.python.keras import backend as k

base = pd.read_csv('iris.csv')

res = base.iloc[:, 4].values
prev = base.iloc[:, 0:4].values
res = LabelEncoder().fit_transform(res)

print(prev)
print(res)

def rede(neurons, activation, kernel_initializer, optimizer, loss):
    k.clear_session()
    classificador = Sequential()
    classificador.add(Dense(units=neurons, activation=activation, input_dim=4, kernel_initializer=kernel_initializer))
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dense(units=3, kernel_initializer=kernel_initializer))
    classificador.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=rede, activation='softmax')

parametros = {'batch_size' : [10, 30],
              'epochs': [1000, 2000],
              'neurons': [4, 8],
              'activation': ['relu', 'tanh'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'optimizer': ['adam', 'sgd'],
              'loss': ['sparse_categorical_crossentropy']}

search = GridSearchCV(estimator=classificador, param_grid=parametros, cv=10)
search = search.fit(prev, res)
melhores_parametros = search.best_params_
melhor_precisao = search.best_score_

print("Melhores Parâmetros:", melhores_parametros)
print("Melhor Precisão:", melhor_precisao)

# Resultado do Tuning 1000 e 2000 epocas: {'activation': 'relu', 'batch_size': 10, 'epochs': 1000, 'kernel_initializer': 'random_uniform', 'loss': 'categorical_crossentropy', 'neurons': 4, 'optimizer': 'adam'}