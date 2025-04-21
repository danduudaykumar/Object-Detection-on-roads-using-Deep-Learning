from flask import Flask, render_template, request, redirect, url_for
import torch
import cv2
import numpy as np
import os
import uuid  # For generating unique filenames

app = Flask(__name__)

# Set upload and output folder
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Load the YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define colors for different object types
object_colors = {
    'car': (0, 255, 0),        # Green
    'bus': (0, 0, 255),        # Red
    'truck': (255, 0, 0),      # Blue
    'motorcycle': (255, 255, 0),  # Cyan
    'bicycle': (255, 165, 0),   # Orange
    'auto': (128, 0, 128),      # Purple
    'person': (255, 192, 203),  # Pink
    'traffic light': (0, 255, 255)  # Yellow
}

@app.route('/')
def index():
    return render_template('two.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    valid_credentials = {"username": "uday", "password": "Uday123"}

    if username == valid_credentials["username"] and password == valid_credentials["password"]:
        return redirect(url_for('welcome'))
    else:
        return render_template("two.html", error="Invalid credentials. Try again.")

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "Error: No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "Error: No selected file", 400

    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    file_ext = file.filename.rsplit('.', 1)[-1].lower()
    if file_ext not in allowed_extensions:
        return "Error: Invalid file format", 400

    # Generate unique filename
    new_filename = f"{uuid.uuid4()}.{file_ext}"
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
    file.save(img_path)

    if model is None:
        return "Error: YOLO model not loaded properly", 500

    # Load and process the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    boxes = results.xyxy[0]  # Bounding boxes
    detected_objects = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        label = results.names[int(cls)] if int(cls) in results.names else "Unknown"

        # Assign color or default to white
        color = object_colors.get(label, (255, 255, 255))

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        detected_objects.append((label, (x1, y1), (x2, y2)))

    # Save output image
    output_filename = f"output_{new_filename}"
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
    cv2.imwrite(output_path, img)

    return render_template('show_results.html', image_path=output_path, detected_objects=detected_objects)

if __name__ == "__main__":
    app.run(debug=True)
