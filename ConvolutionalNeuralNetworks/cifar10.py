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

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


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
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

from keras.models import load_model

model = load_model('best_cifar10.h5')

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('best_cifar10.h5', save_best_only=True, monitor='val_categorical_accuracy', verbose=1, mode='max')
callbacks = [checkpoint]

epochs = 15

hist = model.fit(x_train, y_train, batch_size=128, epochs=epochs, callbacks=callbacks, validation_data=(x_test, y_test), shuffle=True, use_multiprocessing=True)

best = load_model('best_cifar10.h5')
results = best.evaluate(x_test, y_test)

import matplotlib.pyplot as plt

plt.plot(np.arange(0, epochs), hist.history['val_categorical_accuracy'])
plt.show()