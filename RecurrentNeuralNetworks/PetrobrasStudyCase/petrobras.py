import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv("petr4_treinamento.csv")
base = base.dropna()
train = base.iloc[:, 1:7].values

scaler = MinMaxScaler()
train = scaler.fit_transform(train)

x_train = []
y_train = []

for i in range(90, 1242):
    x_train.append(train[i-90:i, :])
    y_train.append(train[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

regressor = Sequential()
regressor.add(LSTM(units=100, input_shape=(x_train.shape[1], 6), return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=100, input_shape=(x_train.shape[1], 6), return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=100, input_shape=(x_train.shape[1], 6), return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=100, input_shape=(x_train.shape[1], 6), return_sequences=False))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation="linear"))
regressor.compile(optimizer="rmsprop", loss="mean_squared_error", metrics=["mean_absolute_error"])

es = EarlyStopping(monitor="loss", min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, verbose=1)
mcp = ModelCheckpoint("best.h5", monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks_list = [es, rlr, mcp]

hist = regressor.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=callbacks_list)

test = pd.read_csv("petr4_teste.csv")
y_test = test.iloc[:, 1:2].values

complete = pd.concat((base, test))
complete = complete.drop("Date", axis=1)

test_inputs = complete[len(complete) - len(test) - 90:]
test_inputs = scaler.transform(test_inputs)

x_test = []
for i in range(90, 112):
    x_test.append(test_inputs[i-90:i, :])
x_test = np.array(x_test)

preds_scaler = MinMaxScaler()
preds_scaler.fit_transform(base.iloc[:, 1:2].values)

best = load_model("best.h5")
preds = best.predict(x_test)

preds = preds_scaler.inverse_transform(preds)

import matplotlib.pyplot as plt

plt.plot(preds, color="blue", label="Predictions")
plt.plot(test["Open"], color="red", label='Actual prices')
plt.xlabel("Time")
plt.ylabel("Prices")
plt.legend()
plt.show()