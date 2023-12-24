import pandas as pd
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import backend as k # atualizado: tensorflow==2.0.0-beta1

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

k.clear_session()
classificador = Sequential([
tf.keras.layers.Dense(units=8, activation = 'relu', kernel_initializer = 'random_uniform', input_dim=30),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(units=8, activation = 'relu', kernel_initializer = 'random_uniform'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(units=1, activation = 'sigmoid')])
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
classificador.fit(previsores, classe, batch_size=10, epochs=100)



classificador_json = classificador.to_json()
with open("breast_cancer.json", 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('breast_cancer_weights.h5')