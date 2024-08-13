import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

# Define paths
input_folder = 'Applied ML Assignment Images/'       # Folder containing the input images
output_folder = 'segmented_images/'  # Folder to save the output images

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize the image segmentation pipeline
pipe = pipeline("image-segmentation", model="google/deeplabv3_mobilenet_v2_1.0_513")

# Function to extract the mask for a specific label (e.g., "person")
def get_mask_for_label(segmented_data, label):
    for segment in segmented_data:
        if segment['label'] == label:
            return segment['mask']
    return None

# Function to process and save segmented images
def process_and_save_images(input_folder, output_folder):
    for image_name in os.listdir(input_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)
            
            # Perform image segmentation
            segmented_data = pipe(image_path)
            
            # Extract the garment mask
            garment_mask = get_mask_for_label(segmented_data, 'person') 
            
            if garment_mask is None:
                print(f"Garment mask not found for {image_name}.")
                continue
            
            # Convert the mask from a PIL image to a NumPy array
            garment_mask = np.array(garment_mask)
            
            # Create a binary mask where the garment is white and the background is black
            binary_mask = (garment_mask > 0).astype(np.uint8) * 255
            
            # Save the binary mask as an image
            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, binary_mask)
            
            print(f"Processed and saved: {output_image_path}")

# Run the processing function
process_and_save_images(input_folder, output_folder)
