from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Define image size expected by the models
photo_size = 224

# Load the saved models
vgg19_model = tf.keras.models.load_model("vgg19_lr_scheduler.h5")
mobilenet_model = tf.keras.models.load_model("mobilenet_lr_scheduler.h5")

# Define weights for the models
vgg19_weight = 0.6
mobilenet_weight = 0.4

# Function to preprocess the image
def preprocess_image(image_path):
    # Load and resize the image
    img = load_img(image_path, target_size=(photo_size, photo_size))
    # Convert image to array and normalize pixel values
    img_array = img_to_array(img) / 255.0
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions using weighted average
def make_weighted_prediction(image_path, vgg19_model, mobilenet_model, vgg19_weight, mobilenet_weight):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    # Make predictions using both models
    vgg19_prediction = vgg19_model.predict(processed_image)[0][0]
    mobilenet_prediction = mobilenet_model.predict(processed_image)[0][0]
    # Combine predictions using weighted average
    weighted_prediction = (vgg19_weight * vgg19_prediction) + (mobilenet_weight * mobilenet_prediction)
    return weighted_prediction

# Function to interpret the prediction
def interpret_prediction(prediction):
    if prediction >= 0.5:
        return "Autistic"
    else:
        return "Not Autistic"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "No selected file"})
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)
            prediction = make_weighted_prediction(file_path, vgg19_model, mobilenet_model, vgg19_weight, mobilenet_weight)
            result = interpret_prediction(prediction)
            return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
