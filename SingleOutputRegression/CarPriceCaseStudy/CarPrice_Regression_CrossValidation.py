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
base = base.drop('abtest', axis=1)
base = base.drop('monthOfRegistration', axis=1)
base = base.drop('powerPS', axis=1)
base = base.drop('brand', axis=1)
base = base.drop('vehicleType', axis=1)

base = base.loc[base['price'] > 400]
base = base.loc[base['price'] < 350000]

base = base.loc[base['yearOfRegistration'] < 2024]
base = base.loc[base['yearOfRegistration'] > 1940]


values = {'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 'fuelType': 'benzin', 'notRepairedDamage': 'nein'}

base = base.fillna(value = values)

X = base.iloc[:, 1:9].values
y = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder   
for i in [1, 2, 4, 5]:
    X[:, i] = LabelEncoder().fit_transform(X[:, i])

from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 2, 4, 5])], remainder='passthrough')
X = transformer.fit_transform(X).toarray()

y = np.reshape(y, (-1, 1))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def network():
    model = Sequential()
    model.add(Dense(units=132, activation='relu', input_dim=264))
    model.add(Dense(units=132, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])
    return model

from keras.wrappers.scikit_learn import KerasRegressor

regressor = KerasRegressor(build_fn=network, epochs=100, batch_size=64)

from sklearn.model_selection import cross_val_score

results = cross_val_score(estimator=regressor, X=X, y=y, cv=10, scoring='r2')
results = results[-4]

mean = results.mean()

std = results.std()

print("Media: ", mean, "\n\nDesvio padrao: ", std)
