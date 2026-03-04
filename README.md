# 🖥 PCB Defect Detection and Classification System

---

## 📌 Project Overview

This project presents a complete automated Printed Circuit Board (PCB) defect detection and classification system using image processing and deep learning techniques.

The system integrates:

- Reference-based image subtraction  
- Contour-based defect localization  
- EfficientNet-B0 transfer learning classification  
- Streamlit-based web interface for real-time prediction  

The objective is to automatically detect, localize, classify, and visualize PCB defects with high accuracy and low inference time.

---

## 🎯 Objectives

- Detect defects by comparing test PCB images with defect-free templates  
- Extract and crop defect regions using contour detection  
- Train a deep learning model for multi-class defect classification  
- Develop a web-based frontend for real-time defect prediction  
- Integrate backend image processing and CNN inference pipeline  
- Export annotated results and prediction logs  

---

## 🗂 Project Structure

```
PCB_Project/
│
├── Milestone_1_Image_Processing/
│   ├── 01_Image_Subtraction.ipynb
│   ├── 02_ROI_Extraction.ipynb
│   └── README.md
│
├── Milestone_2_Model_Training/
│   ├── 03_Model_Training.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   ├── pcb_final_model.pth
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── README.md
│
├── Milestone_3_Web_App/
│   ├── app.py
│   ├── backend/
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Technical Approach

### 🔹 Milestone 1 – Image Processing

- Template matching  
- Absolute image subtraction  
- Adaptive thresholding  
- CLAHE contrast enhancement  
- Morphological filtering  
- Contour detection  
- ROI extraction  

This stage isolates defect regions by comparing test PCB images against reference templates and extracting defect candidates.

---

### 🔹 Milestone 2 – Deep Learning Model

- **Architecture:** EfficientNet-B0 (Transfer Learning)  
- **Framework:** PyTorch  
- **Input Size:** 128 × 128  
- **Optimizer:** Adam  
- **Loss Function:** CrossEntropyLoss  
- Early stopping and learning rate scheduling  
- Confusion matrix and classification report generation  

This milestone focuses on training a robust CNN model for multi-class PCB defect classification.

---

### 🔹 Milestone 3 – Web Application

- **Frontend:** Streamlit  
- **Backend:** Modularized Python pipeline  
- Real-time defect detection  
- Confidence score display  
- Annotated image export  
- CSV prediction log export  

The web application enables users to upload PCB images and receive instant annotated defect predictions.

---

## 📊 Final Model Performance

- ✅ Classification Accuracy: ~99%  
- ✅ Low False Positive Rate  
- ✅ Low False Negative Rate  
- ✅ Stable and repeatable training  
- ✅ Inference time < 3 seconds per image  

The model exceeds the ≥95% accuracy requirement specified in the evaluation criteria.

---

## 🛠 Tech Stack

| Component         | Tools Used |
|------------------|------------|
| Image Processing | OpenCV, NumPy |
| Deep Learning    | PyTorch, Torchvision |
| Dataset          | DeepPCB |
| Frontend         | Streamlit |
| Evaluation       | Scikit-learn, Matplotlib, Seaborn |
| Deployment       | Modular Python Backend |

---

## 🚀 How To Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Web Application

```bash
streamlit run Milestone_3_Web_App/app.py
```

### 3️⃣ Upload Images

- Upload Template PCB Image  
- Upload Test PCB Image  
- View annotated output with defect labels  
- Download annotated image and CSV report  

---

## 📦 Deliverables Completed

- ✔ Defect mask generation  
- ✔ ROI extraction pipeline  
- ✔ Trained CNN model checkpoint  
- ✔ Accuracy and confusion matrix  
- ✔ Web UI with upload functionality  
- ✔ Backend inference pipeline  
- ✔ Annotated output export  
- ✔ CSV prediction log export  

---

## 🏁 Final Outcome

This project delivers a fully functional PCB defect detection and classification system with:

- High accuracy  
- Modular architecture  
- Real-time visualization  
- Professional milestone-based structure  
- Deployment-ready frontend interface  

---

## 👩‍💻 Author

Developed as part of an academic milestone-based project submission for the PCB Defect Detection and Classification System.
