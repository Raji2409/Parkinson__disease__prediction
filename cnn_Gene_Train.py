# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:01:01 2022

@author: okokp
"""

from glob import glob

import pandas as pd
from PIL.ImageFile import ImageFile
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_path = 'datasets/train'
valid_path = 'datasets/test'

# load model without output layer

IMAGE_SIZE = [300, 300]
# add preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# useful for getting number of classes
folders = glob('datasets/train/*')


x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Use the Image Data Generator to import the images from the

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('datasets/train',
                                                 target_size = (300, 300),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

val_set = val_datagen.flow_from_directory('datasets/test',
                                            target_size = (300, 300),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# fit the model
history = model.fit_generator(training_set,
                              validation_data=val_set,
                              epochs=5,
                              steps_per_epoch=len(training_set),
                              validation_steps=len(val_set))

model.save('model_eye.h5')


print(pd.DataFrame(history.history))

pd.DataFrame(history.history).plot()

