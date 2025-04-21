import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
import torch
import cv2
# import numpy as np # Not explicitly used it seems

# --- Application Setup ---
app = Flask(__name__) # Flask automatically finds 'static' and 'templates' folders next to app.py

# --- Corrected Path Definitions ---
# Get the absolute path of the directory where this script ('app.py') is located
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Define the main static folder relative to app.py
STATIC_FOLDER_PATH = os.path.join(BASE_DIR, 'static')

# Define subfolder names *within* the static folder
UPLOAD_SUBFOLDER = 'uploads'
OUTPUT_SUBFOLDER = 'output'

# Create full *file system paths* for saving files inside the static folder
UPLOAD_FOLDER_PATH = os.path.join(STATIC_FOLDER_PATH, UPLOAD_SUBFOLDER)
OUTPUT_FOLDER_PATH = os.path.join(STATIC_FOLDER_PATH, OUTPUT_SUBFOLDER)

# Create the actual directories on the file system if they don't exist
# Ensure your project structure has: video/static/uploads and video/static/output
os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

# --- Load the YOLOv5 model ---
try:
    # Load model once when the app starts for efficiency
    # Use pretrained=True which is standard, force_reload is usually for debugging cache issues
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load YOLOv5 model: {e}")
    model = None # Set model to None if loading fails

# --- Define colors --- (Keep as is)
object_colors = {
    'car': (0, 255, 0), 'bus': (0, 0, 255), 'truck': (255, 0, 0),
    'motorcycle': (255, 255, 0), 'bicycle': (255, 165, 0), 'auto': (128, 0, 128),
    'person': (255, 192, 203), 'traffic light': (0, 255, 255)
}

# --- Routes ---

@app.route('/')
def index():
    # Assumes 'two.html' is your login page in the 'templates' folder
    return render_template('two.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    valid_credentials = {"username": "uday", "password": "Uday123"}

    if username == valid_credentials["username"] and password == valid_credentials["password"]:
        # Assumes 'welcome.html' exists in the 'templates' folder
        return redirect(url_for('welcome'))
    else:
        return render_template("two.html", error="Invalid credentials. Try again.")

@app.route('/welcome')
def welcome():
    # Assumes 'welcome.html' exists in the 'templates' folder
    return render_template('welcome.html')

@app.route('/upload')
def upload():
    # Assumes 'upload.html' exists in the 'templates' folder
    return render_template('upload.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if model loaded correctly at startup
    if model is None:
        print("Error processing request: Model not loaded.")
        return "Error: Object detection model is currently unavailable", 500

    if 'image' not in request.files:
        print("Error: No 'image' file part in request.")
        return "Error: No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        print("Error: No file selected (empty filename).")
        return "Error: No selected file", 400

    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    try:
        # Safely get extension
        file_ext = file.filename.rsplit('.', 1)[1].lower()
    except IndexError:
        print(f"Error: Uploaded file '{file.filename}' has no extension.")
        return "Error: Invalid file format (missing extension)", 400

    if file_ext not in allowed_extensions:
        print(f"Error: File extension '{file_ext}' not allowed.")
        return f"Error: Invalid file format. Allowed types: {', '.join(allowed_extensions)}", 400

    # --- Saving Uploaded File (Corrected) ---
    # Generate unique filename base
    unique_id = str(uuid.uuid4())
    upload_filename = f"{unique_id}.{file_ext}"
    # Use the correct *file system path* variable
    img_save_path = os.path.join(UPLOAD_FOLDER_PATH, upload_filename)
    try:
        file.save(img_save_path)
        print(f"Uploaded image saved to: {img_save_path}")
    except Exception as e:
        print(f"ERROR saving uploaded file to {img_save_path}: {e}")
        return "Error saving uploaded file.", 500

    # --- Load and process the image ---
    try:
        img = cv2.imread(img_save_path)
        if img is None:
            print(f"ERROR: cv2.imread failed to load image from {img_save_path}")
            # Clean up the invalid saved file?
            # os.remove(img_save_path)
            return "Error: Could not read the uploaded image file.", 500

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb) # Perform inference

        # Draw bounding boxes on the *original BGR* image (img)
        boxes = results.xyxy[0]
        # detected_objects = [] # Keep if you intend to display this list in HTML

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box.tolist()
            label = results.names[int(cls)] if int(cls) < len(results.names) else "Unknown"
            color = object_colors.get(label, (255, 255, 255)) # Default to white
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2) # Use thickness 2
            # Display label and confidence score
            display_text = f"{label} {conf:.2f}"
            cv2.putText(img, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # detected_objects.append((label, (x1, y1), (x2, y2))) # Add if needed

    except Exception as e:
        print(f"ERROR during image processing or drawing: {e}")
        return "Error during image processing.", 500

    # --- Save output image (Corrected) ---
    output_filename = f"output_{unique_id}.{file_ext}"
    # Use the correct *file system path* variable
    output_save_path = os.path.join(OUTPUT_FOLDER_PATH, output_filename)
    try:
        success = cv2.imwrite(output_save_path, img)
        if not success:
             print(f"ERROR: cv2.imwrite failed to save image to {output_save_path}")
             return "Error saving processed image.", 500
        print(f"Processed image saved to: {output_save_path}")
    except Exception as e:
        print(f"ERROR saving processed file with cv2.imwrite to {output_save_path}: {e}")
        return "Error saving processed image.", 500

    # --- Prepare Path for Template (Corrected) ---
    # Create the path RELATIVE to the 'static' folder, using FORWARD SLASHES.
    # This is what url_for('static', filename=...) needs.
    image_path_for_template = f"{OUTPUT_SUBFOLDER}/{output_filename}"
    print(f"DEBUG: Path passed to template: {image_path_for_template}")

    # --- Render Template ---
    # Assumes 'show_results.html' exists in the 'templates' folder
    return render_template('show_results.html', image_path=image_path_for_template)
    # return render_template('show_results.html', image_path=image_path_for_template, detected_objects=detected_objects) # If passing list


if __name__ == "__main__":
    print("--- Starting Flask App ---")
    print(f"Base directory: {BASE_DIR}")
    print(f"Static folder path: {STATIC_FOLDER_PATH}")
    print(f"Uploads saving to: {UPLOAD_FOLDER_PATH}")
    print(f"Outputs saving to: {OUTPUT_FOLDER_PATH}")
    print("--------------------------")
    # Use host='0.0.0.0' to make accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)