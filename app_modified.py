from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import io
import os
import cv2

app = Flask(__name__)
CORS(app)

# Static folder for saving output images
OUTPUT_DIR = "/static"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define the model paths and disease labels for different models
MODEL_PATHS = {
    "mango": r"C:\Drive E\me\FruitApp\Web\best_model_fold_2.keras",  # Path to mango model
    "strawberry": r"C:\Drive E\me\FruitApp\Web\best_model_fold_3.keras"  # Path to strawberry model
}

DISEASE_LABELS = {
    "mango": [
        "Alternaria", "Anthracnose", "Bacterial Canker", "Black Mould Rot",
        "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew",
        "Scooty Mould", "Stem End Rot"
    ],
    "strawberry": [
        "Angular Leaf Spot", "Anthracnose Fruit Rot", "Blossom Blight", "Gray Mold",
        "Leaf Spot", "Powdery Mildew (Fruit)", "Powdery Mildew (Leaf)"
    ]
}

# Load models into memory
models = {}
for model_type, model_path in MODEL_PATHS.items():
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        models[model_type] = model
        print(f"Model for {model_type} loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model for {model_type}: {e}")

@app.route('/')
def index():
    return render_template('index_modified.html')

# Optimized function for color-based fruit detection
def detect_fruit(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

    # Apply Gaussian Blur to reduce noise
    blurred_hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Define broader color ranges for red and yellow (to account for different fruit shades)
    lower_red = np.array([0, 100, 80])
    upper_red = np.array([10, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create masks for red and yellow regions
    mask_red = cv2.inRange(blurred_hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(blurred_hsv, lower_yellow, upper_yellow)

    # Combine the masks
    mask = cv2.bitwise_or(mask_red, mask_yellow)

    # Use morphological transformations to clean the mask (remove noise)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)  # Remove small objects

    # Find contours in the cleaned mask to detect the fruit
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are large enough to be a fruit
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Dynamically adjust threshold based on image size
            # Draw the largest contour on the mask (for debugging)
            cv2.drawContours(mask_cleaned, [largest_contour], -1, (255, 0, 0), 2)
            cv2.imwrite('debug_contour.png', mask_cleaned)  # Save the cleaned mask with contour (for debugging)
            return True  # Fruit detected

    return False  # No fruit detected

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
        original_width, original_height = img.size

        # Step 1: Check if a fruit is present using optimized color-based segmentation
        if not detect_fruit(img):
            return jsonify({"error": "No fruit detected in the image."}), 400

        # Resize the image to match model input size (224x224)
        img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0  # Normalize the image data
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Get the model and labels for the selected fruit
        model = models[fruit_type]
        disease_labels = DISEASE_LABELS[fruit_type]

        # Make a prediction using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = disease_labels[predicted_class]
        confidence = float(predictions[0][predicted_class]) * 100

        # Mock bounding box coordinates (for illustration purposes)
        bbox_resized = [50, 50, 150, 150]  # Example coordinates
        # Scale the bounding box to the original image size
        x_min = int(bbox_resized[0] * (original_width / 224))
        y_min = int(bbox_resized[1] * (original_height / 224))
        x_max = int(bbox_resized[2] * (original_width / 224))
        y_max = int(bbox_resized[3] * (original_height / 224))
        bbox_original = [x_min, y_min, x_max, y_max]

        # Draw the bounding box and label on the image
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox_original, outline="red", width=3)
        draw.text((bbox_original[0], bbox_original[1] - 10),
                  f"{predicted_label} ({confidence:.2f}%)", fill="red")

        # Save the processed image with bounding box to the output directory
        output_filename = f"output_{file.filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        img.save(output_path)

        # Respond with the prediction details and image URL
        return jsonify({
            "label": predicted_label,
            "confidence": f"{confidence:.2f}",
            "image_url": f"/static/{output_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
