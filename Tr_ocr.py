## Tr OCR
import numpy as np
from tqdm.auto import tqdm
from Model_loading import processor_tr_ocr, trocr_model
import torch
from PIL import Image



def apply_TRocr(cropped_image,cell_coordinates):
    """
    Apply TrOCR to the given cell coordinates.
    
    Parameters:
    - cell_coordinates: List of coordinates for cropping cells from an image.
    
    Returns:
    - data: A list of extracted text for each cell.
    """
    data = []  # Initialize a list to store data for each row

    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []  # List to store text for the current row
        for cell in row["cells"]:
            # Crop cell out of the image
            cell_image = np.array(cropped_image.crop(cell["cell"]))
            
            
            # Convert the cell image to RGB format for TrOCR processing
            pil_image = Image.fromarray(cell_image).convert("RGB")
            pixel_values = processor_tr_ocr(images=pil_image, return_tensors="pt").pixel_values

            # Generate text predictions using TrOCR model
            generated_ids = trocr_model.generate(pixel_values)
            text = processor_tr_ocr.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if not text:
                text = "NAN"  # Append "NAN" if no text is detected
            
            row_text.append(text)

        # Append the row's text list to the data list
        data.append(row_text)

    return data