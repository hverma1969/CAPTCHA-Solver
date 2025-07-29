

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras
import random


import matplotlib.pyplot as plt

"""### Load dataset"""

train_data_path = './emnist-balanced-train.csv'
test_data_path = './emnist-balanced-test.csv'

train_data = pd.read_csv(train_data_path, header=None)

type(train_data)

train_data.head(10)

print(len(train_data))

class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

"""## Data is flipped"""

# num_classes = len(train_data[0].unique())
num_classes = 49
row_num = 8
print(num_classes)

plt.imshow(train_data.values[row_num, 1:].reshape([28, 28]), cmap='Greys_r')
plt.show()

img_flip = np.transpose(train_data.values[row_num,1:].reshape(28, 28), axes=[1,0]) # img_size * img_size arrays
plt.imshow(img_flip, cmap='Greys_r')

plt.show()

def show_img(data, row_num):
    img_flip = np.transpose(data.values[row_num,1:].reshape(28, 28), axes=[1,0]) # img_size * img_size arrays
    plt.title('Class: ' + str(data.values[row_num,0]) + ', Label: ' + str(class_mapping[data.values[row_num,0]]))
    plt.imshow(img_flip, cmap='Greys_r')

# 10 digits, 26 letters, and 11 capital letters that are different looking from their lowercase counterparts
# num_classes = train_data.values[:, 0].max() + 1
img_size = 28

def img_label_load(path):
    data = pd.read_csv(path)
    data = data[data[0] <= 46]

    num_classes = data[0].max() + 1  # = 47
    labels = keras.utils.to_categorical(data.values[:, 0], num_classes)  # Now this will work
    images = data.values[:, 1:].reshape((-1, 28, 28, 1)) / 255.0  # Normalize
    return images, labels

"""### model, compile"""

model = keras.models.Sequential()

# model.add(keras.layers.Reshape((img_size,img_size,1), input_shape=(784,)))
model.add(keras.layers.Conv2D(filters=12, kernel_size=(5,5), strides=2, activation='relu', 
                              input_shape=(img_size,img_size,1)))
model.add(keras.layers.Dropout(.5))

model.add(keras.layers.Conv2D(filters=18, kernel_size=(3,3) , strides=2, activation='relu'))
model.add(keras.layers.Dropout(.5))

model.add(keras.layers.Conv2D(filters=24, kernel_size=(2,2), activation='relu'))


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=150, activation='relu'))
model.add(keras.layers.Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())

"""### Train"""

# X, y = img_label_load(train_data_path)
data = pd.read_csv(train_data_path)
data = data[data[0] <= 46]
X = data.iloc[:, 1:].values.astype('float32') / 255.0  # pixel data, normalized
y = data.iloc[:, 0].values  # label

print(X.shape)

data_generator = keras.preprocessing.image.ImageDataGenerator(validation_split=.2)
data_generator_with_aug = keras.preprocessing.image.ImageDataGenerator(validation_split=.2,
                                            width_shift_range=.2, height_shift_range=.2,
                                            rotation_range=60, zoom_range=.2, shear_range=.3)

training_data_generator = data_generator.flow(X, y, subset='training')
validation_data_generator = data_generator.flow(X, y, subset='validation')
history = model.fit(training_data_generator, 
                              steps_per_epoch=100, epochs=3, # can change epochs to 10
                              validation_data=validation_data_generator)

test_X, test_y = img_label_load(test_data_path)
test_data_generator = data_generator.flow(test_X, test_y)


model.evaluate(test_data_generator)

"""## Look at some predictions"""

X_test, y_test = img_label_load(test_data_path) # loads images and orients for model

def run_prediction(idx):
    plt.figure()
    show_img(test_data, idx)
    pred = model.predict(X_test[idx].reshape(1, 28, 28, 1))
    print("Predicted Label:", class_mapping[np.argmax(pred)])
    print("Actual Label   :", class_mapping[np.argmax(y_test[idx])])
    plt.show()

for _ in range(1, 10):
    idx = random.randint(0, X_test.shape[0] - 1)
    run_prediction(idx)


import random

for _ in range(1,10):
    idx = random.randint(0, 47-1)
    run_prediction(idx)

show_img(test_data, 123)
np.argmax(y_test[123])

"""## Keras exports"""

with open('model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('./model.h5')

model.save('./model2.h5')
