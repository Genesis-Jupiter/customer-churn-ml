# Customer Churn Prediction using Classical Machine Learning

## 📌 Project Overview
This project implements an **end-to-end classical machine learning pipeline** to predict
customer churn using tabular telecom data.

The objective is not only to achieve good predictive performance, but also to:
- follow correct machine learning practices
- prevent data leakage
- compare models fairly
- interpret model decisions
- extract business insights using unsupervised learning

The project is inspired by and aligned with best practices from *Hands-On Machine Learning*.

---

## 🎯 Problem Statement
Customer churn is a critical business problem where retaining existing customers is
often cheaper than acquiring new ones.

The goal of this project is to predict whether a customer is likely to churn based on:
- service usage
- billing information
- contract details
- customer tenure

This is formulated as a **binary classification problem**.

---

## 🧠 Machine Learning Approach

### Type
- Supervised Learning (Classification)
- Unsupervised Learning (Clustering & Anomaly Detection)

### Key Challenges Addressed
- Class imbalance
- Mixed numerical and categorical features
- Real-world data cleaning issues
- Bias–variance tradeoff
- Model interpretability

---

## 📂 Dataset
- Telecom customer churn dataset
- Tabular, real-world data
- Target variable: `Churn` (Yes / No)

Key preprocessing steps:
- Conversion of malformed numeric columns (`TotalCharges`)
- Stratified train–test split
- Pipeline-based preprocessing

---

## ⚙️ Pipeline Design

A single preprocessing pipeline is used across all models to ensure:
- no data leakage
- reproducibility
- fair model comparison

### Preprocessing
- Numerical features → StandardScaler
- Categorical features → OneHotEncoder
- Combined using ColumnTransformer

---

## 🤖 Models Trained & Compared

The following models were trained and evaluated using **cross-validated ROC-AUC**:

- Logistic Regression (baseline)
- Support Vector Machine (Linear & RBF)
- Decision Tree (regularized)
- Random Forest

---

## 📊 Evaluation Metrics

Due to class imbalance, **accuracy was avoided**.

Primary metrics:
- ROC-AUC (model comparison)
- Precision, Recall, F1-score (error analysis)

Cross-validation was used to assess model stability.

---

## 🔍 Hyperparameter Tuning
- RandomizedSearchCV used for efficient hyperparameter optimization
- Focused on top-performing models (Random Forest, SVM)
- Overfitting analyzed using learning curves

---

## 🧩 Model Interpretability
- Feature importance extracted from tree-based models
- Key drivers of churn identified and analyzed
- Results aligned with domain intuition (tenure, charges, contract type)

---

## 📉 Unsupervised Learning

### Customer Segmentation
- PCA applied for dimensionality reduction
- K-Means clustering used to identify customer segments
- Clusters interpreted using aggregate statistics

### Anomaly Detection
- Isolation Forest used to detect unusual customer behavior
- Identified customers with rare or extreme usage patterns

---

## 🧪 Final Model Evaluation
- Test set used **only once** for final evaluation
- Performance compared with cross-validation results
- No significant overfitting observed

---

## 📁 Project Structure
See the folder structure section below.

---

## 🚀 Key Takeaways
- Demonstrates a complete ML lifecycle
- Emphasizes correctness over shortcuts
- Uses classical ML techniques effectively
- Suitable for real-world tabular data problems

---

## 🔮 Future Improvements
- Cost-sensitive learning
- Threshold optimization based on business costs
- Deployment using a lightweight Streamlit app
- Monitoring model drift over time

---

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 👤 Author
Khushboo  
