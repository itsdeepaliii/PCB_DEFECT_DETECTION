# Milestone 3 – Web Application and Backend Integration

## Objective

This milestone integrates the trained deep learning model into a web-based application for real-time defect detection and visualization.

## Tasks Implemented

- Developed frontend using Streamlit
- Implemented image upload interface for:
  - Template PCB image
  - Test PCB image
- Modularized backend pipeline
- Integrated trained EfficientNet model
- Generated annotated output images
- Enabled CSV prediction log export

## Backend Modules

- `image_processing.py` – Template matching and mask generation
- `classifier.py` – ROI classification using trained model
- `pipeline.py` – Full defect detection pipeline
- `model_loader.py` – Model checkpoint loading

## Features

- Automatic template matching
- Defect localization with bounding boxes
- Confidence score display
- Downloadable annotated image
- Downloadable CSV prediction log
- Processing time < 3 seconds per image

## Result

The system successfully:

- Detects defects in uploaded PCB images
- Classifies defect types accurately
- Provides real-time visualization via web interface
