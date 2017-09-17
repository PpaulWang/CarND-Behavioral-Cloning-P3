
# coding: utf-8

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Cropping2D,Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    image_name = line[0].split('/')[-1]
    file_path = '../data/IMG/' + image_name
    image = cv2.imread(file_path)
    measurement = float(line[3])
    images.append(image)
    measurements.append(measurement)


    image = np.fliplr(image)
    measurement = -measurement
    images.append(image)
    measurements.append(measurement)

X_train = np.array(images)
Y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70,25),(0,0))))
# Five Convolutional Layer
model.add(Convolution2D(24, 5, 5, border_mode='same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
          
# Five Fully-Connected Layer
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
          


model.compile(loss='mse',optimizer='adam')

model.fit(X_train,Y_train,validation_split=0.3,shuffle=True,nb_epoch=5)
print ("ok")
model.save('final.h5')

