import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint

base = pd.read_csv("petr4_treinamento.csv")
base = base.dropna()
train = base.iloc[:, 1:2].values

testBase = pd.read_csv("petr4_teste.csv")
test = testBase.iloc[:, 1:2]
testComplete = pd.concat((base["Open"], test["Open"]), axis=0)
ent = testComplete[len(testComplete) - len(testBase) - 90:].values
ent = ent.reshape(-1, 1)

scaler = MinMaxScaler()
train = scaler.fit_transform(train)

ent = scaler.transform(ent)

x = []
y = []

for i in range(90, 1242):
    x.append(train[i-90:i, 0])
    y.append(train[i, 0])
x, y = np.array(x), np.array(y)
x.reshape((x.shape[0], x.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(x.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=100    , return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=100))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation="linear"))

regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=["mean_absolute_error"])

callbacks_list = [ModelCheckpoint("best.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')]


x_test = []

for i in range(90, 112):
    x_test.append(ent[i-90:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

hist = regressor.fit(x, y, epochs=150, batch_size=32, callbacks=callbacks_list)

best = load_model("best.h5")

preds = best.predict(x_test)
preds = scaler.inverse_transform(preds)

import matplotlib.pyplot as plt

plt.plot(test, color='red', label="Preço Real")
plt.plot(preds, color='blue', label="Previsões")
plt.title("Previsao de preço das ações")
plt.xlabel("Tempo")
plt.ylabel("Valor Yahoo")
plt.legend()
plt.show()