# README: Evaluating Sampling Techniques and Machine Learning Models on Credit Card Fraud Dataset

## Overview
This project investigates the impact of different sampling techniques on the performance of machine learning models for detecting fraudulent credit card transactions. The dataset is highly imbalanced, which necessitates using resampling methods to improve classification performance. Five sampling techniques and five machine learning models were evaluated to determine the best combination for handling imbalanced data.

---

## Dataset
The dataset contains 772 rows and 31 columns, including:
- **Features**: `V1` to `V28` (principal components), `Time`, `Amount`
- **Target**: `Class` (binary: 0 for legitimate transactions, 1 for fraud)

---

## Class Distribution
![image](https://github.com/user-attachments/assets/4daf9dda-9280-4a67-9d91-8dad454c9817)

---

## Resampling Techniques
The following resampling methods were applied to balance the dataset:
1. **Random Undersampling**: Reduces the majority class randomly to match the minority class.
2. **Random Oversampling**: Replicates samples from the minority class to match the majority class size.
3. **SMOTE (Synthetic Minority Oversampling Technique)**: Generates synthetic samples for the minority class.
4. **NearMiss**: Selects a subset of majority class samples based on their distance from minority class samples.
5. **ClusterCentroids**: Under-samples the majority class by replacing it with the centroids of K-means clusters.

---

## Machine Learning Models
The following models were trained and evaluated:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **K-Nearest Neighbors (KNN)**
5. **Support Vector Machine (SVM)**

---

## Results
The performance of each model was evaluated using the accuracy metric. Below is a summary of the best sampling method and highest accuracy for each model:

| Model                   | Best Sampling Method    | Highest Accuracy |
|-------------------------|-------------------------|------------------|
| Decision Tree           | Random Oversampling    | 0.987097         |
| K-Nearest Neighbors     | Random Oversampling    | 0.980645         |
| Logistic Regression     | Random Oversampling    | 0.877419         |
| Random Forest           | Random Oversampling    | 0.993548         |
| Support Vector Machine  | Random Oversampling    | 0.696774         |

---

## Key Findings
1. **Random Oversampling** consistently provided the best results across all models.
2. **Random Forest Classifier** achieved the highest accuracy (99.35%) when combined with Random Oversampling.
3. **Support Vector Machine** performed poorly compared to other models, even with the best sampling method.

---

## How to Run the Code
1. **Dependencies**: Ensure the following Python packages are installed:
   - `pandas`
   - `scikit-learn`
   - `imblearn`

2. **Steps**:
   - Load the dataset.
   - Apply the five resampling techniques.
   - Train and evaluate the five machine learning models using the test dataset.
   - Generate and analyze the accuracy results.

3. **Execution**:
   - Run the provided Python script.
   - The script outputs accuracy scores and a summary table.

---

