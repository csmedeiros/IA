import pandas as pd
import numpy as np
import os
import cv2

cat_train = []

for image_path in os.listdir("training_set/gato"):
    img = cv2.imread('training_set/gato/'+image_path)
    img = cv2.resize(img, (64, 64))
    cat_train.append(img)
cat_train = np.array(cat_train, dtype='uint8')


dog_train = []
for image_path in os.listdir('training_set/cachorro'):
    img = cv2.imread('training_set/cachorro/'+image_path)
    img = cv2.resize(img, (64, 64))
    dog_train.append(img)
dog_train = np.array(dog_train, dtype='uint8')

cat_test = []

for image_path in os.listdir('test_set/gato'):
    img = cv2.imread('test_set/gato/'+image_path)
    img = cv2.resize(img, (64, 64))
    cat_test.append(img)
cat_test = np.array(cat_test, dtype='uint8')

dog_test = []

for image_path in os.listdir('test_set/cachorro'):
    img = cv2.imread('test_set/cachorro/'+image_path)
    img = cv2.resize(img, (64, 64))
    dog_test.append(img)
dog_test = np.array(dog_test, dtype='uint8')

x_train = np.concatenate((dog_train, cat_train))
x_test = np.concatenate((dog_test, cat_test))

y_train = np.concatenate((np.zeros((2000, 1)), np.ones((2000, 1))))
y_test = np.concatenate((np.zeros((500, 1)), np.ones((500, 1))))

x_train = x_train.astype('float64')
x_test = x_test.astype('float64')

x_train /= 255
x_test /= 255

from keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(rescale=1./255, rotation_range=60, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train = datagen_train.flow_from_directory('training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

datagen_test = ImageDataGenerator(rescale=1./255)
test = datagen_test.flow_from_directory('test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras import regularizers

weight_decay = 1e-4
regularizer = regularizers.L2(weight_decay)

model = Sequential()

model.add(Input(shape=(64, 64, 3)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='elu', kernel_regularizer=regularizer))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='elu', kernel_regularizer=regularizer))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='elu', kernel_regularizer=regularizer))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='elu', kernel_regularizer=regularizer))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu', kernel_regularizer=regularizer))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu', kernel_regularizer=regularizer))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(units=128))
model.add(Dense(units=1, activation='sigmoid'))

from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(decay=1e-6), loss='binary_crossentropy', metrics='binary_accuracy')

from keras.callbacks import ModelCheckpoint

callbacks = [ModelCheckpoint('best.h5', monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')]

epochs=60
batch_size = 128

model = load_model('best.h5')

hist = model.fit_generator(train, steps_per_epoch=4000/32, epochs=epochs, callbacks=callbacks, validation_data=test, validation_steps=1000/32)

best = load_model('best.h5')


results = best.evaluate(x_test, y_test)

img = cv2.imread('training_set/gato/cat.353.jpg')

img = cv2.resize(img, (64, 64))

img = img.astype('float64')

img /= 255

img = np.expand_dims(img, axis=0)

p = best.predict(img)

import matplotlib.pyplot as plt

plt.plot(np.arange(0, epochs), hist.history['val_loss'], "blue")
plt.plot(np.arange(0, epochs), hist.history['loss'], "orange")