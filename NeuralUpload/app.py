# Imports..
from flask import Flask, request, flash, redirect, url_for, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
# Import the libaries
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers

UPLOAD_FOLDER = '../NeuralUpload/static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}

# Set Flask config..
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
l33t = os.urandom(24)
app.config['SECRET_KEY'] = l33t
app.config['MAX_CONTENT_LENGTH'] = 0.5 * 1024 * 1024
# Configure the allowed file types.
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the / root. Allowed methods are GET and POST
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method=="POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)

        import numpy as np
        import matplotlib.pyplot as plt
        plt.style.use('fivethirtyeight')

        # Load the data
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Get the image label

        # Get the image classification
        classification = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))

        # Add a pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Add another convolution layer
        model.add(Conv2D(32, (5, 5), activation='relu'))

        # Add another pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))

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
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model = models.load_model('../NeuralUpload/image_classifier.model')

        # Read image
        new_image = plt.imread(f'../NeuralUpload/static/uploads/{filename}')

        # Resize the image
        from skimage.transform import resize
        resized_image = resize(new_image, (32, 32, 3))

        # Get the predictions
        predictions = model.predict(np.array([resized_image]))

        # Sort the predictions from least to greatest
        list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        x = predictions

        # Here I loop trough the 10 labels and filter them from least to greatest
        for i in range(10):
            for j in range(10):
                if x[0][list_index[i]] > x[0][list_index[j]]:
                    temp = list_index[i]
                    list_index[i] = list_index[j]
                    list_index[j] = temp

        # Here I make a function to convert a list into a tuple for later.
        def convert(list):
            return tuple(list)

        # Create 2 lists that will store the labels and the percentages of the 5 most likely outcomes from the uploaded image
        OutcomeLabelList = []
        OutcomePrecentList = []

        # Loop trough the top 5 results and put them into the lists made earlier
        for i in range(5):
            OutcomeLabel = (classification[list_index[i]])
            OutcomePrecent = round(predictions[0][list_index[i]] * 100, 2)
            OutcomeLabelList.append(OutcomeLabel)
            OutcomePrecentList.append(OutcomePrecent)

        # Convert the lists with the convert function made earlier
        convert(OutcomeLabelList)
        convert(OutcomePrecentList)

        # Create 2 tuples to easily store values into variables :)
        (FirstOutcomeLabel, SecondOutcomeLabel, ThirdOutcomeLabel, FourthOutcomeLabel, FifthOutcomeLabel) = OutcomeLabelList
        (FirstOutcomePrecent, SecondOutcomePrecent, ThirdOutcomePrecent, FourthOutcomePrecent, FifthOutcomePrecent) = OutcomePrecentList

        # Return the results
        return render_template("results.html",
                               FirstOutcomeLabel=FirstOutcomeLabel,
                               SecondOutcomeLabel=SecondOutcomeLabel,
                               ThirdOutcomeLabel=ThirdOutcomeLabel,
                               FourthOutcomeLabel=FourthOutcomeLabel,
                               FifthOutcomeLabel=FifthOutcomeLabel,
                               FirstOutcomePrecent=FirstOutcomePrecent,
                               SecondOutcomePrecent=SecondOutcomePrecent,
                               ThirdOutcomePrecent=ThirdOutcomePrecent,
                               FourthOutcomePrecent=FourthOutcomePrecent,
                               FifthOutcomePrecent=FifthOutcomePrecent,
                               filename=filename)
    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == "__main__":
    app.run()