import pandas as pd
import numpy as np

base = pd.read_csv("autos.csv", encoding='ISO-8859-1')

base = base.drop('seller', axis=1)
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
base = base.drop('name', axis=1)
base = base.drop('offerType', axis=1)

base = base.loc[base['price'] > 400]
base = base.loc[base['price'] < 350000]


values = {'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 'fuelType': 'benzin', 'notRepairedDamage': 'nein'}

base = base.fillna(value = values)

X = base.iloc[:, 1:13].values
y = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder   
for i in [0, 1, 3, 5, 8, 9, 10]:
    X[:, i] = LabelEncoder().fit_transform(X[:, i])

from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])], remainder='passthrough')
X = transformer.fit_transform(X).toarray()

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as k
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


def run():
    
    k.clear_session()
    regressor = Sequential()
    regressor.add(Dense(units=158, activation='relu', input_dim=316))
    regressor.add(Dense(units=158, activation='relu'))
    regressor.add(Dense(units=1, activation='linear'))

    regressor.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_logarithmic_error', metrics=['mean_absolute_percentage_error'])
    
    checkpoint= ModelCheckpoint('bestmodel.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    callbacks = [checkpoint]
    
    epochs=10
    
    hist = regressor.fit(X, y, epochs=epochs, batch_size=32, callbacks=callbacks, verbose=1, validation_split=0.2)
    
    plt.plot(np.arange(0, epochs), hist.history['val_loss'])
    
    plt.show()
    
    return regressor

regressor = run()

best = load_model('bestmodel.h5')

results = best.predict(X)

print(results.mean())
print(y.mean())