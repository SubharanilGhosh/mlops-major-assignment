import os
import joblib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
# We assume 'savedmodel.pth' is in the same directory
model_path = 'savedmodel.pth'
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file 'savedmodel.pth' not found. Please run train.py first.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to process the uploaded image
def process_image(image_file):
    # Open the image using PIL (Python Imaging Library)
    img = Image.open(image_file)

    # The Olivetti dataset images are 64x64 pixels, grayscale.
    # We need to convert the uploaded image to match this.

    # 1. Convert to grayscale
    img_gray = img.convert('L')

    # 2. Resize to 64x64
    img_resized = img_gray.resize((64, 64))

    # 3. Convert the image to a numpy array
    img_array = np.array(img_resized)

    # 4. Flatten the 64x64 array into a 1D array (4096 features)
    # This is the format the model expects
    img_flat = img_array.flatten()

    # 5. Rescale pixel values to be between 0 and 1 (like the original data)
    img_scaled = img_flat / 255.0

    # 6. Reshape for a single prediction
    return img_scaled.reshape(1, -1)

# Define the main route ('/')
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        if model is None:
            error = "Model is not loaded. Cannot make predictions."
            return render_template('index.html', error=error)

        # Check if an image was uploaded
        if 'image' not in request.files:
            error = "No image file selected."
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            error = "No image file selected."
            return redirect(request.url)

        if file:
            try:
                # 1. Process the image to get the feature vector
                processed_img = process_image(file)

                # 2. Make a prediction
                pred_class = model.predict(processed_img)

                # The prediction is a number (the person's ID)
                prediction = f"Predicted Person ID: {pred_class[0]}"

            except Exception as e:
                error = f"Error processing image: {e}"

    # Render the HTML page
    return render_template('index.html', prediction=prediction, error=error)

# Run the app
if __name__ == '__main__':
    # Use '0.0.0.0' to make it accessible inside the Docker container
    app.run(debug=True, host='0.0.0.0', port=5000)