#before running yeh code install this ->"pip install ultralytics opencv-python"
#imp hai for the code to run
import cv2
from ultralytics import YOLO
import numpy as np
# Open webcam (0 is default camera; replace with a video file path if desired)
cap = cv2.VideoCapture(0)


# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Pre-trained lightweight model



# Parameters
min_obstacle_area = 0.2  # Fraction of frame area to consider as obstacle

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    # Process detection results
    frame_area = frame.shape[0] * frame.shape[1]
    obstacle_detected = False
    obstacle_position = None

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x, y, w, h = box.xywh[0].cpu().numpy()
            confidence = box.conf.cpu().numpy()
            if confidence > 0.5:  # Confidence threshold
                box_area = w * h
                if box_area / frame_area > min_obstacle_area:
                    obstacle_detected = True
                    obstacle_position = (x + w/2) / frame.shape[1]  # Normalized x-center
                    break

    # Simulate avoidance logic (print instead of moving)
    if obstacle_detected:
        if obstacle_position < 0.4:
            print("Obstacle on left - Moving right")
        elif obstacle_position > 0.6:
            print("Obstacle on right - Moving left")
        else:
            print("Obstacle in center - Moving up")
    else:
        print("No obstacle - Moving forward")

    # Visualize the detection
    annotated_frame = results[0].plot()  # Draw bounding boxes and labels
    cv2.imshow("Webcam Object Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()