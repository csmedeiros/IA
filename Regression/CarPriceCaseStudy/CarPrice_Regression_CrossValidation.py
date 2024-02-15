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

print(base.loc[pd.isnull(base['vehicleType'])])
print(base['vehicleType'].value_counts()) #limousine
print(base.loc[pd.isnull(base['gearbox'])])
print(base['gearbox'].value_counts()) #manuell
print(base.loc[pd.isnull(base['model'])])
print(base['model'].value_counts()) #golf
print(base.loc[pd.isnull(base['fuelType'])])
print(base['fuelType'].value_counts()) #benzin
print(base.loc[pd.isnull(base['notRepairedDamage'])])
print(base['notRepairedDamage'].value_counts()) #nein

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

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def network():
    model = Sequential()
    model.add(Dense(units=158, activation='relu', input_dim=316))
    model.add(Dense(units=158, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])
    return model

from keras.wrappers.scikit_learn import KerasRegressor

regressor = KerasRegressor(build_fn=network, epochs=100, batch_size=128)

from sklearn.model_selection import cross_val_score

results = cross_val_score(estimator=regressor, X=X, y=y, cv=10, scoring='neg_mean_absolute_error')

mean = results.mean()

std = results.std()

print("Media: ", mean, "\n\nDesvio padrao: ", std)