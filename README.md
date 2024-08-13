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
