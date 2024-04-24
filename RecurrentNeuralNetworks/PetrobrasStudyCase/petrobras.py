import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

base = pd.read_csv("petr4_treinamento.csv")
base = base.dropna()
train = base.iloc[:, 1:2].values
high = base.iloc[:, 2:3].values

feature_scaler = MinMaxScaler()

open_ = feature_scaler.fit_transform(train)
high_ = feature_scaler.fit_transform(high)

x_train = []
y_open = []
y_high = []

for i in range(90, 1242):
    x_train.append(open_[i-90:i, 0])
    y_open.append(open_[i, 0])
    y_high.append(high_[i, 0])
x_train, y_open, y_high = np.array(x_train), np.array(y_open), np.array(y_high)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.column_stack((y_open, y_high))

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=100, return_sequences=False))
regressor.add(Dropout(0.3))
regressor.add(Dense(units=2, activation='linear'))

regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=["mean_absolute_error"])

callbacks_list = [ModelCheckpoint('best.h5', monitor='loss', verbose=1, save_best_only=True), EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1, mode='min'), ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1, mode='min')]

hist = regressor.fit(x_train, y_train, epochs=100,  batch_size=32, callbacks=callbacks_list)

test = pd.read_csv("petr4_teste.csv")
open_test = test.iloc[:, 1:2].values
high_test = test.iloc[:, 2:3].values
complete_base = pd.concat((base["Open"], test["Open"]), axis=0)
inputs = complete_base[len(complete_base) - len(test) - 90:].values
inputs = inputs.reshape(-1, 1)
inputs = feature_scaler.transform(inputs)

x_test = []

for i in range(90, 112):
    x_test.append(inputs[i-90:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

best = load_model("best.h5")
preds = best.predict(x_test)
preds = feature_scaler.inverse_transform(preds)

import matplotlib.pyplot as plt

plt.plot(test.iloc[:, 1:2].values, color='red', label="Open Price")
plt.plot(test.iloc[:, 2:3].values, color='black', label="High Price")
plt.plot(preds[:, 0], color='blue', label="Open Predictions")
plt.plot(preds[:, 1], color='orange', label="High Predictions")
plt.xlabel("Time")
plt.ylabel("Yahoo Prices")
plt.legend()
plt.show()