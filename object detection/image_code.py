import torch
import cv2
import numpy as np
# import opencv
from matplotlib import pyplot as plt

# Load the YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to classify traffic light color based on dominant color in the ROI
def classify_traffic_light_color(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV space
    red_range = ((0, 120, 70), (10, 255, 255))
    yellow_range = ((20, 100, 100), (40, 255, 255))
    green_range = ((35, 50, 50), (85, 255, 255))

    # Create masks for each color
    red_mask = cv2.inRange(hsv_roi, red_range[0], red_range[1])
    yellow_mask = cv2.inRange(hsv_roi, yellow_range[0], yellow_range[1])
    green_mask = cv2.inRange(hsv_roi, green_range[0], green_range[1])

    # Count the number of pixels for each color
    red_count = np.sum(red_mask)
    yellow_count = np.sum(yellow_mask)
    green_count = np.sum(green_mask)

    # Determine the dominant color
    if red_count > yellow_count and red_count > green_count:
        return "Red"
    elif yellow_count > red_count and yellow_count > green_count:
        return "Yellow"
    else:
        return "Green"

# Load an image
img_path = r"C:\Users\bunny\Desktop\object detection\img1.webp"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform inference using YOLOv5
results = model(img_rgb)

# Display the detection results
results.show()

# Get bounding boxes and labels
boxes = results.xyxy[0]  # (x1, y1, x2, y2, confidence, class)
labels = results.names

# COCO class ID for 'traffic light'
traffic_light_class_id = 9  # Ensure correct class ID for 'traffic light'

# Loop through detected objects
for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    if int(cls) == traffic_light_class_id:  # Check if it's a traffic light
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Crop ROI for the traffic light
        roi = img_rgb[y1:y2, x1:x2]

        # Classify the traffic light color
        color = classify_traffic_light_color(roi)
        print(f"Detected Traffic Light Color: {color}")

        # Draw the bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, color, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Show final image with detections
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detected Traffic Lights and Colors")
plt.axis('off')
plt.show()