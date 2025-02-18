import streamlit as st
from PIL import Image
import numpy as np
import io
import pandas as pd

import os
import subprocess

try:
    import ultralytics
except ImportError:
    subprocess.run(["pip", "install", "ultralytics"])

# Import your models and utility functions
from TD import TD_model1
from tsr import TSR, get_cell_coordinates_by_row, apply_ocr
from info_det_ocr import info_det_and_ocr
from Tr_ocr import apply_TRocr

# Set page config
st.set_page_config(
    page_title="INDRA OCR",
    page_icon="🧾",
    layout="wide",
)

# Customizing background color and font color
st.markdown("""
    <style>
    body {
        background-color: blue; /* Change to your desired background color */
        color: #333333; /* Change to your desired font color */
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown(
    """
    <style>
    .main-title {
        font-size: 3.5rem; /* Increased font size */
        color: white;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555555;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    <div class="main-title">INDRA OCR: Intelligent Invoice Data Recognition and Automation</div>
    """,
    unsafe_allow_html=True,
)

# Create two columns for layout: one for the file uploader and another for the image and extracted data
col1, col2 = st.columns([1, 3])  # Adjust the ratio to control the width of the columns

with col1:
    # File uploader for image (placed in the left column, smaller button)
    uploaded_file = st.file_uploader(
        "Upload Your Invoice (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

with col2:
    # Display the uploaded image (right column)
    if uploaded_file is not None:
        # If a new image is uploaded, reset the session state
        if 'uploaded_image' not in st.session_state or st.session_state.uploaded_image != uploaded_file:
            st.session_state.uploaded_image = uploaded_file
            st.session_state.processed_data = None
            st.session_state.cropped_image = None
            st.session_state.cell_coordinates = None
            st.session_state.df = None

        # Process the image only if it hasn't been processed yet
        if st.session_state.processed_data is None:
            # Read the image file as a PIL image
            image = Image.open(uploaded_file)

            # Resize the image to make it smaller for display
            image.thumbnail((800, 800))  # Resize to fit within the 800px limit

            # Convert the image to OpenCV format (NumPy array)
            image_arr = np.array(image)

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", width= 500)

            # Run info detection and OCR
            detected_data = info_det_and_ocr(image_arr)
            st.session_state.processed_data = detected_data

            # Run the Table detection model and crop images
            cropped_image = TD_model1(image_arr)

            if isinstance(cropped_image, list):
                cropped_image = cropped_image[0]  # Extract the image from the list

            st.session_state.cropped_image = cropped_image

            # Run Table Structure Recognition (TSR)
            output_image, cells = TSR(cropped_image)

            # Get cell coordinates and perform OCR on table cells
            cell_coordinates = get_cell_coordinates_by_row(cells)
            st.session_state.cell_coordinates = cell_coordinates

            # Use TR OCR for extracting table data
            data = apply_TRocr(cropped_image, cell_coordinates)

            # Store the DataFrame in session state for CSV download
            st.session_state.df = pd.DataFrame(data)

        # Show extracted data if the dataframe is available
        if st.session_state.df is not None:
            st.markdown("### Extracted Data Table:")
            st.dataframe(st.session_state.df)  # Display the dataframe as a table

            # Show download button if the data has been processed
            st.markdown("### Download Extracted Data")
            # Convert DataFrame to CSV and store in-memory buffer
            csv_buffer = io.StringIO()
            st.session_state.df.to_csv(csv_buffer, index=False)

            # Get the CSV data as bytes
            csv_data = csv_buffer.getvalue().encode('utf-8')

            # Create a download button in Streamlit to download the CSV file
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="extracted_table_data.csv",
                mime="text/csv",
            )

# Footer
st.markdown(
    """
    <div class="footer">© 2024 INDRA OCR | Designed for efficient invoice processing</div>
    """,
    unsafe_allow_html=True,
)
