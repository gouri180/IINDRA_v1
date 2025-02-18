import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from Model_loading import processor_tr_ocr, trocr_model

import os

# Define a path relative to the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "Models", "invoice_yolo_100_7classes.pt")


# Load the YOLO model

info_model = YOLO(model_path)



# Function for performing OCR with TrOCR
def ocr_with_transformer(image):
    # Convert the image to RGB and process it for TrOCR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = processor_tr_ocr(images=image_rgb, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    
    # Decode the generated text from the model
    generated_text = processor_tr_ocr.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Main function for detection, cropping, and OCR
def info_det_and_ocr(image_arr, conf_threshold=0.5):
    # Initialize the dictionary to store detected class names and text
    detected_data = {}

    # Perform inference using YOLO
    results = info_model.predict(image_arr)
    
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
    class_names = results[0].names  # Class names dictionary
    
    # Iterate over the detected objects
    for i, box in enumerate(boxes):
        score = scores[i]
        if score < conf_threshold:
            continue  # Skip low confidence detections

        x1, y1, x2, y2 = map(int, box)  # Convert bounding box to integers
        class_id = class_ids[i]
        class_name = class_names[class_id]

        # Crop the detected region
        cropped_region = image_arr[y1:y2, x1:x2]

        # Perform OCR using TrOCR on the cropped region
        detected_text = ocr_with_transformer(cropped_region)

        # Save the detected text into the dictionary with class name as key
        detected_data[class_name] = detected_text

        # Optional: Display cropped image for debugging
        #plt.imshow(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
        #plt.title(f"Detected: {class_name}")
        #plt.show()

    return detected_data

# You can now call this function by passing the image as a numpy array.
# Example:
# extracted_data = info_det_and_ocr(image_arr)
# print(extracted_data)
