import pandas as pd
import numpy as np
from gc import collect

base = pd.read_csv("autos.csv", encoding='ISO-8859-1')

base = base.drop('seller', axis=1)
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
base = base.drop('name', axis=1)
base = base.drop('offerType', axis=1)
base = base.drop('abtest', axis=1)
base = base.drop('monthOfRegistration', axis=1)
base = base.drop('powerPS', axis=1)
base = base.drop('brand', axis=1)

base = base.loc[base['price'] > 400]
base = base.loc[base['price'] < 350000]

base = base.loc[base['yearOfRegistration'] < 2024] 
base = base.loc[base['yearOfRegistration'] > 1940]


values = {'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 'fuelType': 'benzin', 'notRepairedDamage': 'nein'}

base = base.fillna(value = values)

X = base.iloc[:, 1:9].values
y = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder   
for i in [ 0, 2, 3, 5, 6]:
    X[:, i] = LabelEncoder().fit_transform(X[:, i])

from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 2, 3, 5, 6])], remainder='passthrough')
X = transformer.fit_transform(X).toarray()

y = np.reshape(y, (-1, 1))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras import backend as k
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

k.clear_session()
collect()

regressor = Sequential()
regressor.add(InputLayer(input_shape=(272, )))
regressor.add(Dense(units=272, activation='relu'))
regressor.add(Dense(units=272, activation='relu'))
regressor.add(Dense(units=272, activation='relu'))
regressor.add(Dense(units=272, activation='relu'))
regressor.add(Dense(units=272, activation='relu'))
regressor.add(Dense(units=272, activation='relu'))
regressor.add(Dense(units=272, activation='relu'))
regressor.add(Dense(units=272, activation='relu'))
regressor.add(Dense(units=272, activation='relu'))
regressor.add(Dense(units=272, activation='relu'))

regressor.add(Dense(units=1, activation='linear'))

regressor.compile(optimizer=Adam(learning_rate=0.00001), loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])

checkpoint= ModelCheckpoint('bestmodel.h5', monitor='val_loss',verbose=1, save_best_only=True, mode='min')

callbacks = [checkpoint]

epochs=40

regressor = load_model('bestmodel.h5')

hist = regressor.fit(X, y, epochs=epochs, batch_size=64, callbacks=callbacks, verbose=1, validation_split=0.2)

best = load_model('bestmodel.h5')

results = best.predict(X)

print(results.mean())
print(y.mean())

from sklearn.metrics import r2_score

r2 = r2_score(y, results)
print(r2)

plt.subplot(122)
plt.plot(np.arange(0, epochs), hist.history['val_loss'])
plt.title('Car Price Prediction Model')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Logarithmic Error')
plt.show()