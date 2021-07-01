# Import the libaries
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Load the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10

index = 10

# Get the image label

# Get the image classification
classification = ['airplane','car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship' ,'truck']

# Convert the labels into a set of 10 numbers to input into the neural network
y_train_one_hot = tf.keras.utils.to_categorical(y_train)
y_test_one_hot = tf.keras.utils.to_categorical(y_test)

# Print the new labels

# Normalize the pixels to be values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# Create the models architecture
model = Sequential()

# Add the first layer
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))

# Add a pooling layer
model.add(MaxPooling2D(pool_size= (2,2)))

# Add another convolution layer
model.add(Conv2D(32, (5,5), activation='relu'))

# Add another pooling layer
model.add(MaxPooling2D(pool_size= (2,2)))

# Add a flattening layer
model.add(Flatten())

# Add a layer with 1000 neurons
model.add(Dense(1000, activation='relu'))

# Add a drop out layer
model.add(Dropout(0.5))

# Add a layer with 500 neurons
model.add(Dense(500, activation='relu'))

# Add a drop out layer
model.add(Dropout(0.5))

# Add a layer with 250 neurons
model.add(Dense(250, activation='relu'))

# Add a layer with 10 neurons
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer= 'adam',
              metrics= ['accuracy'])

model = models.load_model('image_classifier.model')

# Read image
new_image = plt.imread('./uploads/index.jpg')
img = plt.imshow(new_image)

# Resize the image
from skimage.transform import resize
resized_image = resize(new_image, (32,32,3))

# Get the predictions
predictions = model.predict(np.array([resized_image]))

# Sort the predictions from least to greatest
list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp

# Show the sorted labels in order
print(list_index)

# Print the first 5 predictions
for i in range(5):
    print(classification[list_index[i]], ":", round(predictions[0][list_index[i]] * 100, 2), '%')