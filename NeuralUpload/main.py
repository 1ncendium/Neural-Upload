import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from skimage.transform import resize

def start_Neural():
    # Load data from the dataset in tuples.
    (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

    # Scale the images down
    training_images, testing_images = training_images / 255, testing_images / 255

    # Create class names for the labels
    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    # Load the neural model
    model = models.load_model('image_classifier.model')

    # Load image and convert color scheme from BGR TO RGB
    img = cv.imread('./uploadsicca1.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Resize image
    img = resize(img, (32,32,3))

    # Prediction
    prediction = model.predict(np.array([img]) / 255)

    # Get the index of the maximum value
    index = np.argmax(prediction)
    print(f'Prediction is {class_names[index]}')

    # Sort the predictions from least to greatest
    list_index = [0,1,2,3,4,5,6,7,8,9]
    x = prediction

    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp

    # Show the sorted labels in order
    print(list_index)

    # Print the first 5 predSictions
    for i in range(5):
        print(class_names[list_index[i]], ":", round(prediction[0][list_index[i]] * 100, 2), '%')
start_Neural()