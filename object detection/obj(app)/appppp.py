from flask import Flask, render_template, request, redirect, url_for
import torch
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

# Function to classify traffic light color (unchanged)
def classify_traffic_light_color(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    red_range = ((0, 120, 70), (10, 255, 255))
    yellow_range = ((20, 100, 100), (40, 255, 255))
    green_range = ((35, 50, 50), (85, 255, 255))

    red_mask = cv2.inRange(hsv_roi, red_range[0], red_range[1])
    yellow_mask = cv2.inRange(hsv_roi, yellow_range[0], yellow_range[1])
    green_mask = cv2.inRange(hsv_roi, green_range[0], green_range[1])

    red_count = np.sum(red_mask)
    yellow_count = np.sum(yellow_mask)
    green_count = np.sum(green_mask)

    if red_count > yellow_count and red_count > green_count:
        return "Red"
    elif yellow_count > red_count and yellow_count > green_count:
        return "Yellow"
    else:
        return "Green"

# Assigning colors for different objects
object_colors = {
    'person': (255, 0, 0),  # Blue for person
    'car': (0, 255, 0),  # Green for car
    'bus': (0, 0, 255),  # Red for bus
    'bike': (255, 255, 0),  # Cyan for bike
    'truck': (0, 255, 255),  # Yellow for truck
    'motorcycle': (255, 0, 255),  # Magenta for motorcycle
    'auto': (255, 165, 0)  # Orange for auto (auto-rickshaw)
}

# Function to get a color for each detected object
def get_object_color(label):
    return object_colors.get(label, (0, 255, 0))  # Default to green if label not found

@app.route('/')
def index():
    return render_template('two.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    valid_credentials = {"username": "user123", "password": "password123"}
    
    if username == valid_credentials["username"] and password == valid_credentials["password"]:
        return redirect(url_for('process_image'))
    else:
        return "Invalid credentials", 401

@app.route('/process_image')
def process_image():
    img_path = r"C:\Users\bunny\Desktop\object detection\static\image.jpg"
    img = cv2.imread(img_path)
    
    if img is None:
        return "Error: Image not found", 404
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    boxes = results.xyxy[0]  # bounding box coordinates
    labels = results.names  # class labels
    
    detected_objects = []  # List to store detected objects
    
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        label = labels[int(cls)]
        
        # Get a distinct color for each detected object
        color = get_object_color(label)
        
        # Draw bounding boxes for all detected objects with increased intensity
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)  # Increased line width to 4
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  # Increased font size
        
        detected_objects.append((label, (x1, y1), (x2, y2)))
        print(f"Detected {label} at ({x1},{y1}) to ({x2},{y2})")

    # Check if any objects were detected
    if not detected_objects:
        print("No objects detected in the image.")
    
    output_path = r"static/output_image.jpg"
    cv2.imwrite(output_path, img)
    print(f"Processed image saved to {output_path}")
    
    return render_template('show_results.html', image_path=output_path, detected_objects=detected_objects)

if __name__ == "__main__":
    app.run(debug=True)
