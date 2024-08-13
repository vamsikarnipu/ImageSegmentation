# Garment Segmentation Batch Processing

This project provides a Python script for batch processing images to perform garment segmentation. The script processes all images in a specified input folder, segments the garments, and saves the results to an output folder.

## Project Structure

- `input_images/`: Folder containing the input images for segmentation.
- `segmented_images/`: Folder where the output segmented images will be saved.
- `garment_segmentation_batch.py`: The main script that performs batch processing and saves segmented images.
- `README.md`: This file.

## Requirements

To run the script, you need Python installed along with the following libraries:

- `opencv-python`
- `numpy`
- `matplotlib`
- `transformers`

You can install the necessary libraries using the following command:

```bash
pip install opencv-python numpy matplotlib transformers

How It Works
Load Images: The script reads all images from the specified input folder.

Image Segmentation:

The transformers library's pipeline is used to perform image segmentation with the pre-trained model google/deeplabv3_mobilenet_v2_1.0_513.
The segmentation model identifies the garment in each image.
Extract and Create Binary Mask:

The script extracts the mask corresponding to the garment (e.g., labeled as "person") and creates a binary mask where the garment is white (255) and the background is black (0).
Save Output Images:

The binary masks are saved as images in the specified output folder.
