from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from utils import *
# from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
train_dir='new_data/train_char'
test_dir='new_data/test_char'
height=64
width=64
batch_size=32
num_classes=26
# img=cv2.imread('train/0/train_30_00000.png')
# print(img.shape)
# print_train()
# print_test()



train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.2,
      zoom_range=0.1,
      horizontal_flip=False,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to target height and width.
        target_size=(height, width),
        color_mode='grayscale',
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(height, width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')
model=models.Sequential()
model.add(layers.Conv2D(32,(5,5),padding='valid',activation='relu',input_shape=(64,64,1)))
model.add(layers.Conv2D(64,(5,5),padding='valid',activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(num_classes,activation='softmax'))
for layer in model.layers[:-5]:
	layer.trainable = False

model.summary()
# model.load_weights('models/model_char4.h5')
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
history = model.fit_generator(
      train_generator,
      epochs=1,
      validation_data=validation_generator,
      verbose=1,)
os.makedirs("./models", exist_ok=True)
model.save('./models/model1.h5')
