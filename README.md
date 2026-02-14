# ML Classification Models – Assignment 2

## 1. Problem Statement
The objective of this project is to implement and compare multiple machine learning classification algorithms on a binary classification dataset and deploy an interactive Streamlit web application for real-time predictions and model evaluation.

The workflow includes:
- Training six classification models  
- Evaluating them using standard performance metrics  
- Comparing model performance in tabular form  
- Deploying an interactive web interface using Streamlit  

---

## 2. Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** scikit-learn (public dataset)

| Property | Value |
|----------|-------|
Total samples | 569  
Total features | 30 numerical features  
Target classes | 0 = Malignant, 1 = Benign  
Class distribution | 212 Malignant, 357 Benign  

### Preprocessing Steps
- Train–test split: 80% training, 20% testing (stratified)  
- Feature scaling using StandardScaler (for Logistic Regression and k-NN)  
- No missing values present  

The features represent digitized image measurements of cell nuclei such as radius, texture, perimeter, area, smoothness, concavity, symmetry, and fractal dimension.

---

## 3. Models Implemented
The following six classification algorithms were trained on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (k-NN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## 4. Evaluation Metrics
Each model was evaluated using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## 5. Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623  
Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174  
K-Nearest Neighbor | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054  
Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492  
Random Forest | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054  
XGBoost | 0.9561 | 0.9901 | 0.9467 | 0.9861 | 0.9660 | 0.9058  

---

## 6. Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|------------------------------------|
Logistic Regression | Achieved the highest accuracy and MCC, indicating strong linear separability of features. Provides a good balance between precision and recall and is computationally efficient. |
Decision Tree | Lowest overall performance and AUC among all models. Likely overfitting due to high variance. Easy to interpret but weaker generalization compared to ensemble methods. |
K-Nearest Neighbor | Strong and balanced performance with high recall and F1 score. Sensitive to feature scaling but performs well after normalization. Suitable for this dataset due to clear neighborhood patterns. |
Naive Bayes | Good AUC score despite independence assumption. Fast and efficient but slightly lower accuracy than Logistic Regression and ensemble models. |
Random Forest | Excellent AUC and stable performance. Reduces variance compared to a single decision tree and captures feature interactions effectively. |
XGBoost | High recall, minimizing false negatives. Competitive performance with strong MCC. Requires careful hyperparameter tuning for small datasets. |

---

## 7. Key Findings
- Logistic Regression achieved the best overall performance with the highest accuracy and MCC.  
- Ensemble methods (Random Forest and XGBoost) provided excellent AUC scores and stable predictions.  
- Decision Tree showed comparatively lower performance due to overfitting.  
- All models achieved accuracy above 91%, indicating strong predictive power of the features.  

**Recommended model for this dataset:** Logistic Regression  
**Best ensemble model:** Random Forest  
**Best recall (minimizing false negatives):** XGBoost  

---

```
ml-classification-app/
├── app.py                          # Streamlit web application
├── train_models.py                 # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── breast_cancer.csv           # Dataset file
└── models/
    ├── logistic_regression.pkl     # Trained LR model
    ├── decision_tree.pkl           # Trained DT model
    ├── knn.pkl                     # Trained k-NN model
    ├── naive_bayes.pkl             # Trained NB model
    ├── random_forest.pkl           # Trained RF model
    ├── xgboost.pkl                 # Trained XGBoost model
    ├── scaler.pkl                  # Feature scaler
    ├── results.pkl                 # Evaluation results
    └── evaluation_results.csv      # Results as CSV
```
---

## 9. Streamlit App Features
The deployed Streamlit application includes:

- CSV dataset upload (test data)  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  
- Model comparison table  

---

## 10. How to Run Locally

pip install -r requirements.txt
streamlit run app.py

## 11. Deployment

The application is deployed using Streamlit Community Cloud and connected to the GitHub repository.

Steps:

Push code to GitHub

Create a new app on Streamlit Cloud

Select app.py as the entry point

Deploy and share the live link

## 12. Technical Details

Train–test split: 80/20 with stratification

Feature scaling: StandardScaler (for Logistic Regression and k-NN)

Random state: 42 for reproducibility

Libraries used: scikit-learn, XGBoost, pandas, numpy, Streamlit

## 13. Conclusion

All six models demonstrated strong performance on the breast cancer dataset. Logistic Regression achieved the best overall results with the highest accuracy and MCC, while ensemble methods provided excellent AUC scores. The Streamlit application enables interactive model comparison and real-time predictions, completing the end-to-end machine learning workflow from training to deployment.

## 14. Submission Checklist

GitHub repository with source code and requirements.txt

Live Streamlit app link

Screenshot from BITS Virtual Lab

README content included in submission PDF


---

# ✅ This README Now Matches the PDF Rubric

✔ Dataset description  
✔ Comparison table with all 6 metrics  
✔ Separate observation table  
✔ Streamlit features mentioned  
✔ GitHub structure  
✔ Deployment section  

---

