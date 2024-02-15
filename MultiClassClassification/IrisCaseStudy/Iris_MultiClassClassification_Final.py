import tensorflow.python as tf
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import os

base = pd.read_csv('iris.csv')
prev = base.iloc[:, 0:4].values
res = base.iloc[:, 4].values
res = LabelEncoder().fit_transform(res)
res = np_utils.to_categorical(res)

print(prev)
print(res)

x_train, x_test, y_train, y_test = train_test_split(prev, res, test_size=0.25)

k.clear_session()
model = Sequential()
model.add(Dense(units=4, input_dim=4, activation='relu', kernel_initializer='random_uniform'))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation='relu', kernel_initializer='random_uniform'))
model.add(Dropout(0.2))
model.add(Dense(units=3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=1000, batch_size=10)

previsoes = model.predict(x_test)
previsoes = (previsoes>0.5)

previsoes = [np.argmax(t) for t in previsoes]
y_test = [np.argmax(t) for t in y_test]

from sklearn.metrics import accuracy_score

print(accuracy_score(previsoes, y_test))

if os.path.exists("iris.json") and os.path.exists("iris_weights.h5"):
    json_file = open("iris.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("iris_weights.h5")
    novo = np.array([[3.2, 4.5, 0.9, 1.1]])
    previsao = loaded_model.predict(novo)
    previsao = (previsao > 0.5)
    if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
        print('Iris setosa')
    elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
        print('Iris virginica')
    elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
        print('Iris versicolor')
else:
    model_json = model.to_json()
    with open("iris.json", "w") as file:
        file.write(model_json)
    print("Modelo salvo em iris.json")

    model.save_weights("iris_weights.h5")
    print("Pesos salvos em iris_weights.h5")