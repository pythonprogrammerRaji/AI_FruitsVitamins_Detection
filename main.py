import tensorflow as tf

print("TensorFlow version:", tf._version_)

hello = tf.constant("Hello, Rajeshwari! TensorFlow is working 🎉")
print(hello.numpy())




# Importing required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set path to dataset folder (update this if your path is different)
dataset_path = "dataset"

# Image size and training parameters
img_height = 180
img_width = 180
batch_size = 8

# Preprocessing images (resizing, normalizing)
datagen = ImageDataGenerator(
    rescale=1./255,       # Normalize pixel values (0 to 1)
    validation_split=0.2  # 20% data for validation
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset="training",
    class_mode="categorical"
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset="validation",
    class_mode="categorical"
)

# Building a simple CNN (Convolutional Neural Network)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')  # Output layer
])

# Compile the model (loss function, optimizer, metrics)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training the model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=3,  # You can increase this after testing
    verbose=1
)

# Import additional libraries for prediction
import numpy as np
from tensorflow.keras.preprocessing import image

# img_path = 'dataset/Fruites/Apple/apple1.jpg',

# Load an image from your computer (change this path to any test image you want)
img_path = "dataset/Onion/onion1.jpg"  # ← replace this with your image file name
img = image.load_img(img_path, target_size=(img_height, img_width))

# Convert image to array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize pixel values

# Make prediction
prediction = model.predict(img_array)

# Get class label
predicted_class = train_data.class_indices  # Dictionary mapping class names to numbers
class_labels = list(predicted_class.keys())
predicted_label = class_labels[np.argmax(prediction)]

# Show result
print(f"Predicted category: {predicted_label}")

# to displat the result on webpages

import os
import csv
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

#Upload floder setup
UPLOAD_FLODER = 'uploads'
os.makedirs(UPLOAD_FLODER, exist_ok=True)
app.config['UPLOAD_FLODER'] = UPLOAD_FLODER

#load fruits info from CSV file

def load_fruit_info():
    fruit_info = {}
    with open('fruit_info.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            fruit = row['fruit'].lower()
            fruit_info[fruit] = {
                "color": row['color'],
                "vitamins": row['vitamins']
            }
    return fruit_info

#load the CSV once when app starts
fruit_data = load_fruit_info()

# dummy predication based on file name
def predict_fruit(image_path):
    filename = os.path.basename(image_path).lower()
    for fruit in fruit_data.keys():
        if fruit in filename:
            return fruit
    return "unknown"


# Upload HTML
# upload_page = '''
# <!DOCTYPE html>
# <html>
# <head><title>Fruit Detector</title></head>
# <body>
#     <h2>Upload Fruit Image</h2>
#     <form action="/predict" method="post" enctype="multipart/form-data">
#         <input type="file" name="image" required>
#         <input type="submit" value="Predict">
#     </form>
# </body>
# </html>
# '''

# Result HTML
# result_page = '''
# <!DOCTYPE html>
# <html>
# <head><title>Prediction Result</title></head>
# <body>
#     <h2>Fruit Prediction Result</h2>
#     <p><strong>Fruit Name:</strong> {{ fruit }}</p>
#     <p><strong>Color:</strong> {{ color }}</p>
#     <p><strong>Vitamins:</strong> {{ vitamins }}</p>
#     <a href="/">Try Another</a>
# </body>
# </html>
# '''

# Home route
@app.route('/')
def home():
    return render_template("upload.html")

#Predict route
@app.route('/predict', methods=['POST'])
def predict():
    image = request.files["image"]
    if image:
        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config['UPLOAD_FLODER'], filename)
        image.save(filepath)
        

        predicted_fruit = predict_fruit(filepath)

        info = fruit_data.get(predicted_fruit, {
            "color": "Not Found",
            "vitamins" : "Not Available"
        })

        #Create the image URl (so the image can be displayed on webpage)
        image_filename = f"/uploads/{filename}"

        return render_template("result.html",
                                      fruit=predicted_fruit,
                                      color=info["color"],
                                      vitamins=info["vitamins"],
                                      image_filename=image_filename)
    else:
        return "No image uploaded!"
    
    #Run server

if __name__ == '__main__':
    print("Flask app is starting...")
    app.run(debug=True)





# ----------------------------------------------------------
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Fruit Classifier</title>
# </head>
# <body>
#     <h2>Upload a Fruit Image</h2>
#     <form action="/predict" method="POST" enctype="multipart/form-data">
#         <input type="file" name="image" required>
#         <input type="submit" value="Predict">
#     </form>
# </body>
# </html>