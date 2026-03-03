# Milestone 2 – Model Training and Evaluation

## Objective

This milestone focuses on training and evaluating a deep learning model to classify PCB defects into predefined categories.

## Tasks Implemented

- Preprocessing ROI images (resized to 128x128)
- Data augmentation (flip, rotation, normalization)
- Implementation of EfficientNet-B0 using PyTorch
- Training using Adam optimizer
- Cross-entropy loss function
- Validation and early stopping
- Performance evaluation

## Model Details

- Architecture: EfficientNet-B0 (Transfer Learning)
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Image Size: 128x128
- Framework: PyTorch

## Performance Metrics

- Test Accuracy: **~99%**
- Confusion Matrix generated
- Classification Report (Precision, Recall, F1-score)

## Output Files

- `pcb_final_model.pth` – Trained model checkpoint
- `confusion_matrix.png`
- `classification_report.txt`

## Result

The trained model achieved high classification accuracy with low false positive and false negative rates, meeting the ≥95% accuracy requirement.
