import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to classify traffic light color
def classify_traffic_light_color(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges
    red_range = ((0, 120, 70), (10, 255, 255))
    yellow_range = ((20, 100, 100), (40, 255, 255))
    green_range = ((35, 50, 50), (85, 255, 255))

    # Create masks
    red_mask = cv2.inRange(hsv_roi, red_range[0], red_range[1])
    yellow_mask = cv2.inRange(hsv_roi, yellow_range[0], yellow_range[1])
    green_mask = cv2.inRange(hsv_roi, green_range[0], green_range[1])

    # Count pixels
    red_count, yellow_count, green_count = np.sum(red_mask), np.sum(yellow_mask), np.sum(green_mask)

    if red_count > yellow_count and red_count > green_count:
        return "Red"
    elif yellow_count > red_count and yellow_count > green_count:
        return "Yellow"
    else:
        return "Green"

# Function to process video
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    traffic_light_class_id = 9  # COCO class ID for 'traffic light'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        boxes = results.xyxy[0]

        for box in boxes:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])
            class_name = results.names[int(cls)]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if int(cls) == traffic_light_class_id:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    light_color = classify_traffic_light_color(roi)
                    cv2.putText(frame, f"Traffic Light: {light_color}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)

        cv2.imshow("Frame", frame)  # Use imshow to show frame in VS Code

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for key press and break on 'q'
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()  # Close any OpenCV windows
    print(f"Processed video saved as: {output_path}")

# Run object detection on the video
input_video_path = r"C:\Users\bunny\Desktop\object detection\video.mp4"  # Update the path as needed
output_video_path = "output_video.mp4"

process_video(input_video_path, output_video_path)
