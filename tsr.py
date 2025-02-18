import torch
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import torch
from ultralytics import YOLO
from torchvision import transforms
from transformers import TableTransformerForObjectDetection
from PIL import ImageDraw
import numpy as np
import csv
import easyocr
from tqdm.auto import tqdm
import csv


device = "cuda" if torch.cuda.is_available() else "cpu"
# new v1.1 checkpoints require no timm anymore
structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
structure_model.to(device)
print("")

from torchvision import transforms

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# update id2label to include "no object"
id2label = structure_model.config.id2label
id2label[len(structure_model.config.id2label)] = "no object"


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def TSR(cropped_image):
    pixel_values = structure_transform(cropped_image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    print(pixel_values.shape)


    # forward pass
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    # update id2label to include "no object"
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, cropped_image.size, structure_id2label)
    #print(cells)

    cropped_table_visualized = cropped_image.copy()
    draw = ImageDraw.Draw(cropped_table_visualized)

    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")

    return cropped_table_visualized , cells
############# Visualizing rows and columns on cropped image


## Modified

def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        # Use the row's Y coordinates for the cell's top and bottom
        cell_ymin = row['bbox'][1]
        cell_ymax = row['bbox'][3]  # Adjust as needed for better height

        # Use the column's X coordinates for the cell's left and right
        cell_xmin = column['bbox'][0]
        cell_xmax = column['bbox'][2]

        return [cell_xmin, cell_ymin, cell_xmax, cell_ymax]

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    # Debugging: Print intermediate results
    #for i, row_info in enumerate(cell_coordinates):
     #   print(f"Row {i}: {row_info['row']}, Cell Count: {row_info['cell_count']}")
      #  for cell in row_info['cells']:
       #     print(f"  Cell Bounding Box: {cell['cell']}, Column Bounding Box: {cell['column']}")

    return cell_coordinates


# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory

def apply_ocr(cell_coordinates,cropped_image):
    # Initialize a list to store data for each row
    data = []

    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []  # List to store text for the current row
        for cell in row["cells"]:
            # Crop cell out of the image
            cell_image = np.array(cropped_image.crop(cell["cell"]))
            # Apply OCR
            result = reader.readtext(np.array(cell_image))
            if len(result) > 0:
                # Extract and join the detected text
                text = " ".join([x[1] for x in result])
                row_text.append(text)
            else:
                row_text.append("NAN")  # Append empty string if no text is detected

        # Append the row's text list to the data list
        data.append(row_text)

    return data


# Print the extracted text for each row
"""for idx, row_data in enumerate(data):
    print(f"Row {idx + 1}: {row_data}")"""

def op_csv(data):

# Define the output CSV file path
    output_csv_file = 'extract.csv'

# Write the data to a CSV file
    try:
        with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            for row in data:
                writer.writerow(row)  # Write each row individually

        print(f"Data successfully written to {output_csv_file}")
        return output_csv_file
    except Exception as e:
        print(f"An error occurred: {e}")



