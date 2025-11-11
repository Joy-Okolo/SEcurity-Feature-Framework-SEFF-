# SEcurity-Feature-Framework-SEFF-

## Predictive Vulnerability Detection using Machine Learning

This project builds a machine learning model to detect whether source code is vulnerable or secure, and to classify both the vulnerability type and corresponding Security Features Framework (SEFF) criteria.  
It uses TF-IDF text features, security-specific keyword indicators, and a Random Forest classifier.

## Overview
Core Steps:
1. Load & Clean Data – from `vulnerability_dataset.csv`, removing quotes and formatting errors.  
2. Encode Categorical Features – for language, vulnerability type, SEFF criteria, and binary labels.  
3. Feature Extraction – combines:
   - TF-IDF vectorization (token-based)
   - Custom flags for `password`, `api_key`, `token`, and `http://`
4. Handle Class Imbalance – using SMOTE for balanced training.  
5. Train Models – three Random Forest classifiers:
   - Secure/Vulnerable Detection  
   - Vulnerability Type Classification  
   - SEFF Criteria Prediction  
6. Evaluate – accuracy, precision, recall, F1-score, and detailed classification report.

## Example Output
=== Secure/Vulnerable Detection ===
Accuracy: 0.94
Precision: 0.95
Recall: 0.94
F1-Score: 0.94

## Dependencies
Install required libraries:
`bash
pip install pandas scikit-learn imbalanced-learn numpy

## How to Run
Update the CSV path in the script:
data = pd.read_csv("path/to/vulnerability_dataset.csv", delimiter=';')

Run the script:
python vulnerability_detection.py

View printed evaluation results for:
Secure/Vulnerable Detection
Vulnerability Type Classification
SEFF Criteria Prediction
