# End-to-End Brain Tumor Classification App

This repository contains an end-to-end application for brain tumor classification. The application uses HTML and CSS for the front end, Flask for the backend, OpenCV for image preprocessing, and a VGG19 model for classification.

## Description

The Brain Tumor Classification App allows users to upload medical images and classify them as either tumor or non-tumor using a pre-trained VGG19 model. The front end of the application is built with HTML and CSS, providing a user-friendly interface. The backend, developed using Flask, handles image uploads, preprocessing with OpenCV, and classification with the VGG19 model.

## Features

- **User-Friendly Interface**: Built with HTML and CSS.
- **Image Upload**: Users can upload medical images for classification.
- **Image Preprocessing**: Utilizes OpenCV for preprocessing uploaded images.
- **Deep Learning Model**: Uses a pre-trained VGG19 model for classification.
- **Real-Time Results**: Provides immediate classification results.


## Requirements

- Python 3.x
- Flask
- OpenCV
- TensorFlow/Keras (for VGG19 model)
- HTML/CSS

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/shreyash18g/Brain_Tumor_Classification_app.git
    cd Brain_Tumor_Classification_app
    ```

2. Install the required dependencies:

    ```bash
    pip install flask opencv-python tensorflow
    ```

3. Download the pre-trained VGG19 model weights and place them in the appropriate directory.

4. Run the Flask application:

    ```bash
    python app.py
    ```

5. Open your web browser and navigate to `http://127.0.0.1:5000/`.



## Usage

1. **Home Page**: The home page allows users to upload an image for classification.
2. **Upload Image**: Users can select and upload a medical image.
3. **Image Preprocessing**: The image is preprocessed using OpenCV.
4. **Classification**: The pre-trained VGG19 model classifies the image as either tumor or non-tumor.
5. **Results**: The classification result is displayed on the results page.



## How it Works

1. **Frontend (HTML & CSS)**: Provides a clean and intuitive interface for users to upload images and view results.
2. **Backend (Flask)**: Manages image uploads, preprocessing, and classification logic.
3. **Image Preprocessing (OpenCV)**: Processes the uploaded images to the required format for the VGG19 model.
4. **Classification (VGG19 Model)**: Uses a pre-trained VGG19 model to classify the processed images.

## Contributing

Contributions to this project are welcome! Feel free to open issues or submit pull requests.


## Acknowledgments

- This project utilizes the VGG19 model for classification.
- Special thanks to the contributors of Flask, OpenCV, and TensorFlow/Keras libraries.


