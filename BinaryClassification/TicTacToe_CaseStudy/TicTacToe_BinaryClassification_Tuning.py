import pandas as pd
import numpy as np

raw_data = pd.read_csv('tic-tac-toe.data')

raw_data.head()

raw_data.shape

raw_data['positive'] = raw_data['positive'].map({'negative':0, 'positive':1})

raw_data.columns

raw_data = pd.get_dummies(data=raw_data, columns=['x', 'x.1', 'x.2', 'x.3', 'o', 'o.1', 'x.4', 'o.2', 'o.3'], dtype=int)

raw_data.head()

raw_data.shape

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras import backend as k
from sklearn.model_selection import cross_val_score


result = raw_data['positive']

result = pd.DataFrame(data=result, columns=['positive'])

result.head()

df = raw_data.drop('positive', axis=1)


x_train, x_test, y_train, y_test = train_test_split(df, result, test_size=0.25, random_state=120)

k.clear_session()

def network_creation(optimizer, loss, activation, kernel_initializer, neurons):
    network = Sequential()
    network.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=27))
    network.add(Dropout(0.2))
    network.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    network.add(Dropout(0.2))
    network.add(Dense(units=1, activation='sigmoid'))
    network.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return network

classificador = KerasClassifier(build_fn=network_creation)

parametros = {'batch_size': [10, 30], 
              'epochs': [50, 100],
              'optimizer': [keras.optimizers.Adam(decay=0.0001), keras.optimizers.SGD(learning_rate=0.001, decay=0.0001)],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}

grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy')
grid_search = grid_search.fit(df, result)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_