from flask import Flask, render_template, request, url_for
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model("fruit_model.h5")

# Load CSV with vitamin and color info
fruit_info = pd.read_csv("fruit_info.csv")

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to predict fruit from uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    # Save uploaded file to upload folder
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load image and preprocess
    img = image.load_img(filepath, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)

    if confidence < 40:
        predicted_fruit = "Not Confident",
        color = "Not Avaiable",
        vitamin = "Not Avaiable",
        food_type = "Unknown"

    


    # Define the same order of labels used in model training
    class_labels = ['apple', 'banana', 'carrot', 'mango', 'onion', 'potato', 'rice', 'tomato', 'wheat']

    if predicted_index < len(class_labels):
        predicted_fruit = class_labels[predicted_index]
    else:
        predicted_fruit = "Unknown"

    # Get color and vitamin from CSV
    row = fruit_info[fruit_info['fruit'].str.lower() == predicted_fruit.lower()]
    if not row.empty:
        color = row['color'].values[0]
        vitamin = row['vitamin'].values[0]
        food_type = row['type'].values[0] if 'type' in row.columns else "Unknown"
    else:
        color = "Not available"
        vitamin = "Not available"
        food_type = "Unknown"

    return render_template('result.html',
                           fruit=predicted_fruit,
                           color=color,
                           vitamin=vitamin,
                           food_type=food_type,
                           confidence = confidence,
                           image_filename=filename if filename else "default.jpg")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)