
# coding: utf-8

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('../behave-clone/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    image_name = line[0].split('/')[-1]
    file_path = '../behave-clone/IMG/' + image_name
    image = cv2.imread(file_path)
    images.append(image)
    measurements.append(float(line[3]))

X_train = np.array(images)
Y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')

model.fit(X_train,Y_train,validation_split=0.2,shuffle=True,nb_epoch=10)

model.save('model.h5')

