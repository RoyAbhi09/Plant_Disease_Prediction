from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model
model_path = 'model/Plant_Disease_Classification.h5'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")

# Function to preprocess image
def load_and_prep_image(image_path, target_size=(210, 210)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr / 255.0
    return img_arr


class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the image
            image = load_and_prep_image(filepath, target_size=(210,210))

            # Predict
            prediction = model.predict(image)
            prediction_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction))

            # Prepare response
            prediction_result = {
                'prediction': class_names[prediction_class],
                'confidence': confidence
            }

            return render_template('index.html', prediction=prediction_result, image_path=filename)
        except Exception as e:
            print(f"Error processing image or making prediction: {e}")
            return render_template('index.html', error=f"Error processing image or making prediction: {e}")

    return render_template('index.html', error='Unknown error occurred')

if __name__ == '__main__':
    app.run(debug=True)
