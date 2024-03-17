import numpy as np
import pandas as pd

import keras.datasets.cifar10 as cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=90,
                 width_shift_range=0.1, height_shift_range=0.1,
                 horizontal_flip=True)
datagen.fit(x_train)

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout
from keras import regularizers

weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Flatten())

model.add(Dense(10, activation='softmax'))

from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0006), loss='categorical_crossentropy', metrics='categorical_accuracy')

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('best.h5', monitor='val_categorical_accuracy', mode='max', save_best_only=True, verbose=1)

callbacks = [checkpoint]

epochs = 125

hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), steps_per_epoch=x_train.shape[0]/128, epochs=epochs, callbacks=callbacks, validation_data=(x_test, y_test))

from keras.models import load_model

best = load_model('best.h5')

results = best.evaluate(x_test, y_test)

results2 = best.evaluate(x_train, y_train)

print('Accuracy = ', results[1])

import matplotlib.pyplot as plt

plt.subplot(121)
plt.plot(np.arange(0, epochs), hist.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Val Categorical Accuracy')

plt.subplot(122)
plt.plot(np.arange(0, epochs), hist.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Categorical Accuracy')
plt.title('Cifar10 Model Analisys')

plt.show()

model_json = model.to_json()
with open('model_cifar10.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('weights.h5')