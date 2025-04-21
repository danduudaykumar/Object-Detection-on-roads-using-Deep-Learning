from flask import Flask, render_template, request, redirect, url_for
import torch
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

# COCO class labels mapping with corresponding colors
coco_labels = {
    1: ("Bicycle", (255, 0, 0)), 2: ("Car", (0, 255, 0)), 3: ("Motorcycle", (0, 0, 255)),
    5: ("Bus", (255, 255, 0)), 7: ("Truck", (255, 0, 255)), 9: ("Traffic Light", (0, 255, 255)),
    10: ("Fire Hydrant", (128, 0, 128)), 13: ("Stop Sign", (255, 165, 0)),
    14: ("Parking Meter", (75, 0, 130)), 15: ("Person", (0, 128, 128)),
    24: ("Auto", (255, 140, 0))  # Added Auto (Three-wheeler / Rickshaw)
}

# Function to classify traffic light color
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

# Function to detect all objects on the road
def detect_objects(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    boxes = results.xyxy[0]
    labels = results.names
    
    detected_objects = []
    
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls = int(cls)
        class_info = coco_labels.get(cls, (labels.get(cls, "Unknown"), (255, 255, 255)))
        class_name, color = class_info
        detected_objects.append((class_name, x1, y1, x2, y2))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)  # Increased line width
        cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    return frame, detected_objects

@app.route('/')
def index():
    return render_template('two.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    valid_credentials = {"username": "user123", "password": "password123"}
    
    if username == valid_credentials["username"] and password == valid_credentials["password"]:
        return redirect(url_for('process_video'))
    else:
        return "Invalid credentials", 401

@app.route('/process_video')
def process_video():
    video_path = r"C:\Users\bunny\Desktop\object detection\video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return "Error: Video not found", 404
    
    output_path = r"static/output_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, detected_objects = detect_objects(frame)
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"Processed video saved to {output_path}")
    return render_template('show_results.html', video_path=output_path)

if __name__ == "__main__":
    app.run(debug=True)
