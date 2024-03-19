import numpy as np
import cv2

from keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.3, horizontal_flip=True, rescale=1./255)
datagen_test = ImageDataGenerator(rescale=1./255)

train = datagen_train.flow_from_directory('training_set', target_size=(64, 64), class_mode='binary', batch_size=10)
test = datagen_test.flow_from_directory('test_set', target_size=(64, 64), class_mode='binary', batch_size=10)

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Dropout, Input

model = Sequential()

model.add(Input(shape=(64, 64, 3)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='elu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',activation='elu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(units=4, activation='elu'))
model.add(Dense(units=4, activation='elu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

callbacks_list = [ModelCheckpoint('best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

hist = model.fit_generator(train, epochs=100, validation_data=test, callbacks=callbacks_list, steps_per_epoch=196/10, validation_steps=73/10)

best = load_model("best.h5")

result = best.evaluate(test)

print("Percentual de acerto: {}%".format(result[1]*100))

import matplotlib.pyplot as plt

plt.plot(np.arange(0, 100), hist.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Val Loss")

plt.plot(np.arange(0, 100), hist.history['loss'], color='orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()