import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw
import tensorflow as tf
import io
import os
from ultralytics import YOLO

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Force PyTorch to use CPU
device = torch.device('cpu')

app = Flask(__name__)
CORS(app)

# Static folder for saving output images
OUTPUT_DIR = os.path.join(os.getcwd(), "static")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define the model paths and disease labels for different models
MODEL_PATHS = {
    "mango": r"Models\Mango.keras",
    "strawberry": r"Models\Strawberry.keras"
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
        with tf.device('/cpu:0'):  # Force model to load on CPU
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        models[model_type] = model
        print(f"Model for {model_type} loaded successfully on CPU.")
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        raise RuntimeError(f"Failed to load model for {model_type}: {e}")

# Load YOLO model using CPU
try:
    yolo_model = YOLO(os.path.join(os.getcwd(), "Models", "best - small.pt"))
    yolo_model.to('cpu')  # Ensure YOLO model is on CPU
    print("YOLO model loaded successfully on CPU.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    raise RuntimeError(f"Failed to load YOLO model: {e}")

# List of valid fruit names
valid_fruits = ['mango', 'strawberry']

# List of class names for your YOLO model
class_names = [
    'apple', 'avocado', 'banana', 'blueberry', 'chico', 'custard apple', 
    'dragonfruit', 'grape', 'guava', 'kiwi', 'mango', 'No Fruit', 'orange', 
    'papaya', 'pineapple', 'pomegranate', 'raspberry', 'strawberry', 'watermelon'
]

@app.route('/')
def index():
    return render_template('index.html')

def predict_fruit_with_yolo(img):
    # Convert PIL Image to NumPy array for YOLO
    img = np.array(img)

    # Resize image to 640x640 as required by YOLO model
    img_resized = cv2.resize(img, (640, 640))

    # Convert to BGR (as expected by OpenCV)
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    # Perform inference with YOLO on CPU
    with torch.no_grad():  # Disable gradient calculation for inference
        results = yolo_model(img_bgr)  # YOLO inference on CPU

    # Extract predictions from results
    pred = results[0].boxes

    # Get the class IDs and confidence scores
    class_ids = pred.cls.cpu().numpy()
    confidences = pred.conf.cpu().numpy()

    # Access class names directly from the YOLO model
    class_labels = yolo_model.names

    if len(class_ids) == 0:
        raise ValueError("No fruits detected in the image.")

    # Get the highest confidence class
    highest_confidence_index = np.argmax(confidences)
    predicted_class = class_labels[int(class_ids[highest_confidence_index])]
    confidence = confidences[highest_confidence_index]

    if predicted_class.lower() not in valid_fruits:
        raise ValueError(f"The image that you uploaded has {predicted_class} in it. Please upload a valid image.")

    return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    fruit_type = request.form.get('fruit_type', '').lower()
    if fruit_type not in models:
        return jsonify({"error": "Invalid fruit type provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Predict fruit type using YOLO on CPU
        predicted_fruit, confidence = predict_fruit_with_yolo(img)

        if predicted_fruit.lower() != fruit_type:
            return jsonify({"error": f"The uploaded image is of a {predicted_fruit}, not a {fruit_type}."}), 400
        
        original_width, original_height = img.size
        if fruit_type == "mango":
            input_size = (224, 224)
        elif fruit_type == "strawberry":
            input_size = (384, 384)
        
        img_resized = img.resize(input_size)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Perform disease classification on CPU
        with tf.device('/cpu:0'):
            predictions = models[fruit_type].predict(img_array)
        
        disease_labels = DISEASE_LABELS[fruit_type]
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = disease_labels[predicted_class]
        disease_confidence = float(predictions[0][predicted_class]) * 100

        # Mock bounding box
        bbox_resized = [50, 50, 150, 150]
        x_min = int(bbox_resized[0] * (original_width / input_size[0]))
        y_min = int(bbox_resized[1] * (original_height / input_size[1]))
        x_max = int(bbox_resized[2] * (original_width / input_size[0]))
        y_max = int(bbox_resized[3] * (original_height / input_size[1]))
        bbox_original = [x_min, y_min, x_max, y_max]

        # Draw bounding box and label
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox_original, outline="red", width=3)
        draw.text((bbox_original[0], bbox_original[1] - 10),
                  f"{predicted_label} ({disease_confidence:.2f}%)", fill="red")

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
    app.run(debug=True, host='0.0.0.0', port=os.getenv('PORT', 10000))
