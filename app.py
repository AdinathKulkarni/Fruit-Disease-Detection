import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session
from markupsafe import Markup
from flask_cors import CORS
from PIL import Image, ImageDraw
import tensorflow as tf
import io
import os
import requests
import logging
from dotenv import load_dotenv
from ultralytics import YOLO
import re

# Load environment variables from .env file
load_dotenv()

# Set up logging for production (less verbose)
logging.basicConfig(level=logging.INFO)  # Changed to INFO for better debugging

# Load API key and URL from environment variables for Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")

# Load Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Load OpenAI API key as fallback for audio transcription
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_API_KEY or not GEMINI_API_URL:
    raise ValueError("Gemini API key or URL is missing. Please set them in the .env file.")

if not GROQ_API_KEY and not OPENAI_API_KEY:
    logging.warning("Neither GROQ_API_KEY nor OPENAI_API_KEY is set. Voice transcription will not work.")

# Disable GPU and use only CPU for TensorFlow and PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow
torch.set_num_threads(1)  # Ensure it uses only the CPU by limiting the number of threads

app = Flask(__name__)
app.secret_key = '1234'  # Add any random secret key here
CORS(app)

# Static folder for saving output images
OUTPUT_DIR = os.path.join(os.getcwd(), "static")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define the model paths and disease labels for different models
MODEL_PATHS = {
    "mango": r"Models/Mango.keras",  # Path to mango model
    "strawberry": r"Models/Strawberry_improved.keras",  # Path to strawberry model
    "apple": r"Models/Apple.keras"  # Path to apple model
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
    ],
    "apple": ["Blotch Apple", "Healthy Apple", "Rot Apple", "Scab Apple"]
}

# Load disease classification models into memory
models = {}
for model_type, model_path in MODEL_PATHS.items():
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        models[model_type] = model
    except Exception as e:
        raise RuntimeError(f"Failed to load model for {model_type}: {e}")

# Load YOLO model using the correct method (ultralytics package)
yolo_model = YOLO(r'Models/best - small.pt')  # Path to your YOLO model

# List of valid fruit names
valid_fruits = ['mango', 'strawberry', 'apple']

# List of class names for your YOLO model
class_names = [
    'apple', 'avocado', 'banana', 'blueberry', 'chico', 'custard apple', 
    'dragonfruit', 'grape', 'guava', 'kiwi', 'mango', 'No Fruit', 'orange', 
    'papaya', 'pineapple', 'pomegranate', 'raspberry', 'strawberry', 'watermelon'
]

def predict_fruit_with_yolo(img):
    # Convert PIL Image to NumPy array for YOLO
    img = np.array(img)

    # Resize image to 640x640 as required by YOLO model
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

# Updated function to format Gemini responses with proper HTML formatting
def get_gemini_response(user_input):
    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': GEMINI_API_KEY,
    }
    
    # Use a known valid model identifier 
    model_id = 'gemini-2.0-flash'  # A verified model name
    
    # Build the complete URL - use the exact format from Google's documentation
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    
    # Enhanced prompt with validation logic and more detailed structure
    structured_prompt = f"""
    First, evaluate if the following query is about a plant or fruit disease: "{user_input}"
    
    If the query is NOT about plant or fruit diseases, agricultural pests, crop conditions, or plant health, 
    respond ONLY with: "Please ask a question related to plant or fruit diseases, agricultural pests, or crop conditions."
    
    If the query IS about plant or fruit diseases, provide a comprehensive analysis with the following structure:
    
    <strong>Disease Identification:</strong><br>
    - Full name and scientific classification
    - Causative agent (fungus, bacteria, virus, etc.)
    - Common varieties/strains
    - Geographic distribution
    <br><br>
    
    <strong>Symptoms & Detection:</strong><br>
    - Visual indicators (with detailed description)
    - Progression stages
    - How to differentiate from similar diseases
    - Detection methods for farmers
    <br><br>
    
    <strong>Management & Treatment:</strong><br>
    - Preventive measures
    - Treatment Suggestion
    <br><br>
    
    <strong>Additional Resources:</strong><br>
    - Key information for farmers
    <br><br>
    
    Format your response using the HTML tags shown above. Make your explanation detailed but accessible to farmers.
    """
    
    payload = {
        "contents": [
            {
                "role": "user", 
                "parts": [{"text": structured_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2048  # Increased token limit for more detailed responses
        }
    }

    try:
        # Send request to the Gemini API
        response = requests.post(
            api_url, 
            json=payload, 
            headers=headers
        )
        
        if response.status_code == 200:
            response_data = response.json()
            text_response = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No content returned.")
            return text_response
        else:
            return f"<strong>Error:</strong> Unable to get response from the AI model. Status code: {response.status_code}"
    except Exception as e:
        return f"<strong>Error:</strong> {str(e)}"

def clean_gemini_response(response_text):
    # Remove ```html and ``` from the response
    cleaned_text = re.sub(r'```html\s*|```', '', response_text)
    # Remove extra new lines or unwanted spaces if needed
    cleaned_text = cleaned_text.strip()
    return cleaned_text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask a question', methods=['GET', 'POST'])
def queries():
    if request.method == 'POST':
        # Get user input from the form
        user_query = request.form.get('user_query')
        if user_query:
            # Get the Gemini response based on the user query
            gemini_response = get_gemini_response(user_query)
            # Mark the response as safe HTML so it renders properly
            gemini_response = Markup(gemini_response)
            gemini_response = clean_gemini_response(gemini_response)

            return render_template('query.html', gemini_response=gemini_response, user_query=user_query)
    # For GET requests, render the template without any response
    return render_template('query.html')

@app.route('/voice_query', methods=['POST'])
def voice_query():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']

    # Save the audio file temporarily to disk
    temp_audio_path = os.path.join(OUTPUT_DIR, "temp_audio.webm")
    audio_file.save(temp_audio_path)

    try:
        # Groq API endpoint
        groq_api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

        # Set the headers
        headers = {
    'Authorization': f'Bearer {GROQ_API_KEY}'
}

# Prepare the file and data
        with open(temp_audio_path, 'rb') as f:
            files = {
                'file': ('audio.webm', f, 'audio/webm')
            }

            # Allow multiple languages (English, Hindi, and Marathi)
            data = {
                'model': 'whisper-large-v3',
                # 'language': 'hi, mr, en'  # Hindi, Marathi, English
                # 'language': 'hi',  # For Hindi
                # 'language': 'mr',  # For Marathi
                # 'language': 'en'  # For English

            }

            # Send the request to Groq API
            response = requests.post(groq_api_url, headers=headers, files=files, data=data)

            if response.status_code == 200:
                response_data = response.json()
                transcription = response_data.get('text', '')
            else:
                transcription = f"Error: {response.text}"

    except Exception as e:
            transcription = f"Exception occurred: {str(e)}"

    finally:
            # Remove the temporary audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    return jsonify({'transcription': transcription})



# Add a route to clear the form and response
@app.route('/clear_query')
def clear_query():
    # Clear the session variable storing the response
    session.pop('gemini_response', None)
    return redirect('/ask a question')

@app.route('/contact')
def contact():
    return render_template('contact.html')

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
        if fruit_type == "mango":
            input_size = (224, 224)
        elif fruit_type == "strawberry":
            input_size = (224, 224)
        elif fruit_type == "apple":
            input_size = (224, 224)
        
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

        # Get a natural language response from Gemini API for disease prediction with HTML formatting
        gemini_input = f"The fruit is a {predicted_fruit} with a disease prediction of {predicted_label} at {disease_confidence:.2f}% confidence."
        gemini_response = get_gemini_response(gemini_input)

        # Create a fruit-specific folder in the static directory if it doesn't exist
        fruit_folder = os.path.join(OUTPUT_DIR, predicted_fruit)
        if not os.path.exists(fruit_folder):
            os.makedirs(fruit_folder)

        # Save output image in the fruit-specific folder, named after the disease
        output_filename = f"{predicted_label.replace(' ', '_')}_{file.filename}"
        output_path = os.path.join(fruit_folder, output_filename)
        img.save(output_path)

        return jsonify({
            "label": predicted_label,
            "confidence": f"{disease_confidence:.2f}",
            "image_url": f"/static/{predicted_fruit}/{output_filename}",
            "gemini_response": gemini_response  # HTML formatted Gemini response
        })

    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=4000)