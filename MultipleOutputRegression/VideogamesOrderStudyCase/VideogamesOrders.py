import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, Input, Activation
from keras.models import Model
from keras.optimizers import Adam

base = pd.read_csv("games.csv")

base = base.drop("Other_Sales", axis=1)
base = base.drop("Global_Sales", axis=1)
base = base.drop("Developer", axis=1)

base = base.dropna(axis=0)

gameName = base.Name
base = base.drop("Name", axis=1)

base = base.loc[base["NA_Sales"] > 1]
base = base.loc[base["NA_Sales"] > 1]

X = base.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
na = base.iloc[:, 4].values
eu = base.iloc[:, 5].values
jp = base.iloc[:, 6].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])
X[:, 8] = le.fit_transform(X[:, 8])

                             
transformer = ColumnTransformer(transformers=[("OneHotEncoder", OneHotEncoder(), [0, 2, 3, 8])], remainder='passthrough')
X = transformer.fit_transform(X).toarray()

na = np.reshape(na, (-1, 1))
eu = np.reshape(eu, (-1, 1))
jp = np.reshape(jp, (-1, 1))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
na = scaler.fit_transform(na)
eu = scaler.fit_transform(eu)
jp = scaler.fit_transform(jp)

inputLayer = Input(shape=(74, ))
hiddenLayer1 = Dense(units=32, activation='sigmoid')(inputLayer)
hiddenLayer2 = Dense(units=32, activation='sigmoid')(hiddenLayer1)
hiddenLayer3 = Dense(units=32, activation='sigmoid')(hiddenLayer2)
hiddenLayer4 = Dense(units=32, activation='sigmoid')(hiddenLayer3)
outputLayer1 = Dense(units=1, activation='linear')(hiddenLayer4)
outputLayer2 = Dense(units=1, activation='linear')(hiddenLayer4)
outputLayer3 = Dense(units=1, activation='linear')(hiddenLayer4)

regressor = Model(inputs=inputLayer, outputs=[outputLayer1, outputLayer2, outputLayer3])
regressor.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('bestmodel.h5', save_best_only=True, monitor='loss')

callbacks = [checkpoint]

epochs = 10000

hist = regressor.fit(X, [na, eu, jp], epochs=epochs, batch_size=128, callbacks=callbacks) 

from keras.models import load_model

regressor = load_model('bestmodel.h5')
predNA, predEU, predJP = regressor.predict(X)

import matplotlib.pyplot as plt

plt.plot(np.arange(0, epochs), hist.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.show()

from sklearn.metrics import r2_score

r2NA, r2EU, r2JP = r2_score(na, predNA), r2_score(eu, predEU), r2_score(jp, predJP)
print(r2NA)
print(r2EU)
print(r2JP)