import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Input

import cv2

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#plt.imshow(x_train[0], cmap ='gray')
#plt.title('Classe '+str(y_train[0]))

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train/=255
x_test/=255

y_test = np_utils.to_categorical(y_test, 10)
y_train = np_utils.to_categorical(y_train, 10)
    
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='categorical_accuracy')

from keras.callbacks import ModelCheckpoint

callbacks = [ModelCheckpoint('best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
hist = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test), callbacks=callbacks)
results = model.evaluate(x_test, y_test)

from keras.models import load_model

best = load_model('best.h5')

import random

i = random.randint(0, 9999)
img = x_test[i]
cv2.imshow("img", img)
img = np.expand_dims(img, axis=0)

p = best.predict(img)
pResult = np.argmax(p)

import matplotlib.pyplot as plt

plt.subplot(121)
plt.plot(np.arange(0, 5), hist.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Val Loss")

plt.subplot(122)
plt.plot(np.arange(0, 5), hist.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()
