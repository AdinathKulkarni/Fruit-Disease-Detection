import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw
import tensorflow as tf
import io
import os
from ultralytics import YOLO  # Import the correct YOLO model loader

app = Flask(__name__)
CORS(app)

# Static folder for saving output images
OUTPUT_DIR = os.path.join(os.getcwd(), "static")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define the model paths and disease labels for different models
MODEL_PATHS = {
    "mango": r"Models\Mango.keras",  # Path to mango model
    "strawberry": r"Models\Strawberry.keras"  # Path to strawberry model
}

DISEASE_LABELS = {
    "mango": [
        "Alternaria", "Anthracnose", "Bacterial Canker", "Black Mould Rot",
        "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew",
        "Scooty Mould", "Stem End Rot"
    ],
    "strawberry": [
        "Angular Leaf Spot", "Anthracnose Fruit Rot", "Blossom Blight", "Gray Mold", "Healthy",
        "Leaf Spot", "Powdery Mildew"
    ]
}

# Load disease classification models into memory
models = {}
for model_type, model_path in MODEL_PATHS.items():
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        models[model_type] = model
        print(f"Model for {model_type} loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model for {model_type}: {e}")

# Load YOLO model using the correct method (ultralytics package)
yolo_model = YOLO(r'Models\best.pt')  # Path to your YOLO model

# List of valid fruit names (only mango and strawberry are valid)
valid_fruits = ['mango', 'strawberry']

# List of class names for your YOLO model
class_names = [
    'apple', 'avocado', 'banana', 'blueberry', 'chico', 'custard apple', 
    'dragonfruit', 'grape', 'guava', 'kiwi', 'mango', 'No Fruit', 'orange', 
    'papaya', 'pineapple', 'pomegranate', 'raspberry', 'strawberry', 'watermelon'
]

@app.route('/')
def index():
    return render_template('index_modified.html')

def predict_fruit_with_yolo(img):
    # Convert PIL Image to NumPy array for YOLO
    img = np.array(img)

    # Resize image to 320x320 as required by YOLO model
    img_resized = cv2.resize(img, (640, 640))

    # If your image is in RGB, convert it to BGR (as expected by OpenCV)
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    # Perform inference with YOLO
    results = yolo_model(img_bgr)  # YOLO expects a NumPy array of images

    # Extract predictions from results
    pred = results[0].boxes  # First result in the batch

    # Get the class IDs and confidence scores
    class_ids = pred.cls.cpu().numpy()  # YOLO class IDs
    confidences = pred.conf.cpu().numpy()  # YOLO confidence scores

    # Access class names directly from the YOLO model
    class_labels = yolo_model.names

    if len(class_ids) == 0:
        raise ValueError("No fruits detected in the image.")

    # Get the highest confidence class
    highest_confidence_index = np.argmax(confidences)
    predicted_class = class_labels[int(class_ids[highest_confidence_index])]
    confidence = confidences[highest_confidence_index]

    # Ensure the predicted class is in the list of valid fruits
    if predicted_class.lower() not in valid_fruits:
        raise ValueError(f"The image that you uploaded has {predicted_class} in it. Please upload a valid image.")

    return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Get the fruit type from request parameters (e.g., "mango" or "strawberry")
    fruit_type = request.form.get('fruit_type', '').lower()
    if fruit_type not in models:
        return jsonify({"error": "Invalid fruit type provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load the image from the file
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Predict the fruit type using YOLO
        predicted_fruit, confidence = predict_fruit_with_yolo(img)

        # Check if the YOLO prediction matches the selected fruit
        if predicted_fruit.lower() != fruit_type:
            return jsonify({"error": f"The uploaded image is of a {predicted_fruit}, not a {fruit_type}."}), 400
        
        # Resize the image for disease classification
        original_width, original_height = img.size
        if fruit_type == "mango":
            input_size = (224, 224)
        elif fruit_type == "strawberry":
            input_size = (384, 384)
        
        img_resized = img.resize(input_size)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Load the disease classification model for the selected fruit
        model = models[fruit_type]
        disease_labels = DISEASE_LABELS[fruit_type]

        # Predict disease class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = disease_labels[predicted_class]
        disease_confidence = float(predictions[0][predicted_class]) * 100

        # Mock bounding box (for illustration)
        bbox_resized = [50, 50, 150, 150]
        x_min = int(bbox_resized[0] * (original_width / input_size[0]))
        y_min = int(bbox_resized[1] * (original_height / input_size[1]))
        x_max = int(bbox_resized[2] * (original_width / input_size[0]))
        y_max = int(bbox_resized[3] * (original_height / input_size[1]))
        bbox_original = [x_min, y_min, x_max, y_max]

        # Draw the bounding box and disease label
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox_original, outline="red", width=3)
        draw.text((bbox_original[0], bbox_original[1] - 10),
                  f"{predicted_label} ({disease_confidence:.2f}%)", fill="red")

        # Save output image
        output_filename = f"output_{file.filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        img.save(output_path)

        return jsonify({
            "label": predicted_label,
            "confidence": f"{disease_confidence:.2f}",
            "image_url": f"/static/{output_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=4000)
