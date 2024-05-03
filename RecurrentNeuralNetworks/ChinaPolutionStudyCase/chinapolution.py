import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

base = pd.read_csv("poluicao.csv")
todrop = ["No", "year", "month", "day", "hour", "cbwd"]
for feature in todrop:
    base = base.drop(feature, axis=1)
base = base.dropna().values

train = base[0:int(len(base)*0.8), 0:7]
test = base[33406:41757, 0]
test = test.reshape(-1, 1)

scaler = MinMaxScaler()
train = scaler.fit_transform(train)

x_train = []
y_train = []

for i in range(20, 33405):
    x_train.append(train[i-20:i, 0:7])
    y_train.append(train[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=100, return_sequences=False))
regressor.add(Dropout(0.3))
regressor.add(Dense(units=1, activation="linear"))
regressor.compile(optimizer="rmsprop", loss="mean_squared_error", metrics=["mean_absolute_error", "mean_absolute_percentage_error"])

callbacks_list = [ModelCheckpoint("best.h5", monitor="loss", verbose=1, save_best_only=True, mode="min"), EarlyStopping(monitor="loss", min_delta=1e-10, patience=10, verbose=1, mode="min"), ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, verbose=1)]

hist = regressor.fit(x_train, y_train, batch_size=128, epochs=100, callbacks=callbacks_list)

best = load_model('best.h5')

inputs = base[len(base)-len(test)-90:, :]
inputs = scaler.fit_transform(inputs)

x_test = []

for i in range(90, 8441):
    x_test.append(inputs[i-90:i, :])
x_test = np.array(x_test)

preds = best.predict(x_train)

scaler.fit_transform(base[:, 0:1])

preds = scaler.inverse_transform(preds)

import matplotlib.pyplot as plt

plt.plot(base[:, 0], color="red", label="Preço real")
plt.plot(preds, color="blue", label="Preço previsto")
plt.xlabel("Horas")
plt.ylabel("pm2.5")
plt.legend()
plt.show()