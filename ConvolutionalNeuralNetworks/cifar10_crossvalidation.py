import numpy as np
import pandas as pd

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

X = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

y = np.argmax(y, axis=1)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

def network():
    
    model = Sequential()

    model.add(Conv2D(32, (5,5), input_shape=(32, 32, 3), activation='relu', ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5,5), activation='relu', ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model
    
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

estimator = KerasClassifier(build_fn=network, epochs=15, batch_size=128)
results = cross_val_score(estimator=estimator, X=X, y=y, scoring='accuracy', cv=4, verbose=1)

mean = results.mean()
std = results.std()

print(results)
print(mean)
print(std)