# Breast Cancer Prediction using KNN

This project predicts whether a breast tumor is malignant or benign using
the K-Nearest Neighbors (KNN) algorithm on the Breast Cancer Wisconsin dataset.

## Dataset
- Source: sklearn.datasets.load_breast_cancer
- Features: 30 numerical features
- Target labels:
  - 0 → Malignant
  - 1 → Benign

## Workflow
1. Load dataset
2. Split into training and testing sets
3. Standardize features using StandardScaler
4. Train KNN classifier
5. Evaluate model performance
6. Predict result for a new patient

## Model Details
- Algorithm: K-Nearest Neighbors (KNN)
- Neighbors (k): 5
- Accuracy: ~95%

## How to Run
```bash
pip install -r requirements.txt
python knn_model.py
