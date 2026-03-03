# Milestone 1 – Dataset Preparation and Image Processing

## Objective

This milestone focuses on preprocessing PCB images and extracting defect regions using classical image processing techniques.

## Tasks Implemented

- Dataset setup and inspection (DeepPCB dataset)
- Template and test image alignment
- Reference-based image subtraction
- Adaptive thresholding for defect mask generation
- Contour detection using OpenCV
- ROI (Region of Interest) extraction for defect regions

## Techniques Used

- OpenCV
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Absolute Difference
- Adaptive Gaussian Thresholding
- Morphological Operations
- Contour Detection & Bounding Boxes

## Outputs

- Defect masks highlighting faulty regions
- Cropped defect ROIs
- Visualization images with bounding boxes

## Result

The system successfully:

- Detected key defect regions
- Generated accurate bounding boxes
- Extracted clean ROI samples for model training

This milestone prepared structured defect data for deep learning classification in Milestone 2.
