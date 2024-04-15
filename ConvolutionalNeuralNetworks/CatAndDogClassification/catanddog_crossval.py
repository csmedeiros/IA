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

x = np.concatenate((x_train, x_test))

y_train = np.concatenate((np.zeros((2000, 1)), np.ones((2000, 1))))
y_test = np.concatenate((np.zeros((500, 1)), np.ones((500, 1))))

y = np.concatenate((y_train, y_test))

x = x.astype('float64')

x /= 255

batch_size=128

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=5)
resultados = []

from keras import regularizers

weight_decay = 1e-4
regularizer = regularizers.L2(weight_decay)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True)
datagen.fit(x)

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input


epochs = 15

for index_training, index_testing in kfold.split(x, y):
    
    #print('Training Indexes: ', index_training, 'Testing index', index_testing)    
 
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
    
    model.add(Dense(units=128, activation='elu'))
    model.add(Dense(units=128, activation='elu'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics='binary_accuracy')
    model.fit(x[index_training], y[index_training], batch_size=batch_size, epochs=epochs)
    acc = model.evaluate(x[index_testing], y[index_testing])
    resultados.append(acc[1])