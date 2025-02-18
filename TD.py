import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

import os

# Define a path relative to the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "Models", "TD_3k_yolo_150_wandb.pt")


## Load the YOLO model
model1 = YOLO(model_path)

def TD_model1(image_arr, padding=5, confidence_threshold=0.5):
    # Convert to RGB (for PIL cropping)
    image_rgb = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model1.predict(image_arr)

    # Check if any tables were detected
    if len(results[0].boxes) == 0:
        return []  # No tables detected

    # Extract bounding boxes and confidence scores
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Filter predictions by confidence threshold
    valid_boxes = []
    for i, box in enumerate(boxes):
        if confidences[i] >= confidence_threshold:
            valid_boxes.append((box, confidences[i]))

    # If no valid boxes, return an empty list
    if not valid_boxes:
        return []

    # Sort the boxes by confidence in descending order
    valid_boxes = sorted(valid_boxes, key=lambda x: x[1], reverse=True)

    # Crop the image based on the valid bounding boxes
    cropped_images = []
    image_pil = Image.fromarray(image_rgb)  # Convert NumPy array to PIL image

    for i, (box, conf) in enumerate(valid_boxes):
        x1, y1, x2, y2 = box
        # Add padding to the bounding box coordinates (with boundary checks)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_pil.width, x2 + padding)
        y2 = min(image_pil.height, y2 + padding)

        cropped_image = image_pil.crop((x1, y1, x2, y2))  # Crop using the bounding box coordinates
        cropped_images.append(cropped_image)
    
    return cropped_images




