# Convolutional Neural Network
# Importing the libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

# Check TensorFlow version
tf.__version__

# Part 1 - Data Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('datasets/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_set = test_datagen.flow_from_directory('datasets/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters=32, padding="same", kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
classifier.add(MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
classifier.add(Conv2D(filters=32, padding='same', kernel_size=3, activation='relu'))
classifier.add(MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=128, activation='relu'))

# Step 5 - Output Layer
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.summary()

# Part 3 - Training the CNN

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
classifier.fit(x = training_set, validation_data = test_set, epochs = 10)

# save it as a h5 file
classifier.save('model_breast.h5')

# load model
model = load_model('model_breast.h5')

model.summary()

from matplotlib import pyplot as plt


girls_me = [8, 11, 20, 30, 47, 55, 69, 70, 76, 74, 80, 87]
girls_friend = [16, 12, 11, 14, 15, 11, 8, 7, 6, 7, 8, 6]

yticks = [10, 20, 30, 40, 50, 60, 70, 80]
months = ["0", "1", "2", "4", "5", "6", "7", "8", "9", "10", "11", "12"]


x_axis = months
y_axis = girls_me


plt.figure(figsize=(8, 6)) 
axis = plt.subplot(1, 1, 1) 
                           

# Plotting data on graph
plt.plot(x_axis, y_axis, color="red", marker="o", linestyle="--")
plt.plot(x_axis, girls_friend, color="green", marker="o", linestyle="--")
axis.set_yticks(yticks)


plt.legend(["Accuracy", "Loss"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy for Epoch")



# Showing Figure
plt.show()

# Part 4 - Making a single prediction

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('datasets/2.png', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0]<=0.5:
    print("The image classified is apple ")
elif result[0]>=0.5:
     print("The image classified is Mango")
else:
    print("The image classified is none")

# For multi class problems few changes are needed

# Step 5 - Output Layer: Change units based on class and activation function as softmax
#cnn_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the CNN: Change loss function as categorical_crossentropy
# cnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])