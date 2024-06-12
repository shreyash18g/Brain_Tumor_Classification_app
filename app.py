# Import necessary libraries

import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained VGG19 model
base_model = VGG19(include_top=False, input_shape=(240,240,3))

# Build your classification layers on top of the VGG19 base
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)

# Create your model
model_03 = Model(base_model.inputs, output)

# Load the pre-trained weights
weights_path = r"C:\Users\icham\OneDrive\Desktop\vgg_unfrozen (1).h5"
model_03.load_weights(weights_path)

# Print a message indicating that the model is loaded
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    """Function to get class name based on class number"""
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


def getResult(img):
    """Function to preprocess image and get prediction"""
    # Read and preprocess the image
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((240, 240))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    
    # Get prediction from the model
    result = model_03.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01


@app.route('/', methods=['GET'])
def index():
    """Render the index.html template"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    """Endpoint to handle image upload and prediction"""
    if request.method == 'POST':
        # Get the uploaded file
        f = request.files['file']
        
        # Save the file securely
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Get prediction and class name
        value = getResult(file_path)
        result = get_className(value)
        return result
    
    # Return None if method is not POST
    return None


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
