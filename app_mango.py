from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import io
import os

app = Flask(__name__)
CORS(app)

# Static folder for saving output images
OUTPUT_DIR = "static"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load your pre-trained model
MODEL_PATH = r"C:\Drive E\me\UBUNTU\DenseNet_For_Mango\best_model_fold_3.keras"  # Replace with your model's path
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


@app.route('/')
def index():
    # Render index.html from the templates folder
    return render_template('index_mango.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load the original image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        original_width, original_height = img.size

        # Resize the image for model input
        img_resized = img.resize((224, 224))  # Resize to match model input size
        img_array = np.array(img_resized) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Define disease labels for your dataset
        disease_labels = [
                "Alternaria",
                "Anthracnose",
                "Bacterial Canker",
                "Black Mould Rot",
                "Cutting Weevil",
                "Die Back",
                "Gall Midge",
                "Healthy",
                "Powdery Mildew",
                "Scooty Mould",
                "Stem End Rot",
            ]
        predicted_label = disease_labels[predicted_class]
        confidence = float(predictions[0][predicted_class]) * 100

        # Mock bounding box coordinates from the model (scaled for 224x224 input)
        bbox_resized = [50, 50, 150, 150]  # Example: [x_min, y_min, x_max, y_max]

        # Scale the bounding box back to the original image size
        x_min = int(bbox_resized[0] * (original_width / 224))
        y_min = int(bbox_resized[1] * (original_height / 224))
        x_max = int(bbox_resized[2] * (original_width / 224))
        y_max = int(bbox_resized[3] * (original_height / 224))
        bbox_original = [x_min, y_min, x_max, y_max]

        # Draw bounding box and label on the original image
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox_original, outline="red", width=3)
        draw.text((bbox_original[0], bbox_original[1] - 10),
                  f"{predicted_label} ({confidence:.2f}%)", fill="red")

        # Save the image with the bounding box
        output_filename = f"output_{file.filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        img.save(output_path)

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
