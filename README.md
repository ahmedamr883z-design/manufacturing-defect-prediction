# 🏭 Manufacturing Defect Prediction

ML model that predicts manufacturing defects based on production and material data, with a Streamlit app for real-time predictions and confidence scores.

## 📊 Overview
This project uses historical manufacturing data (production volume, material cost, maintenance hours, quality scores, etc.) to predict whether a production batch is likely to have defects — helping guide material sourcing and quality decisions.

## 🔍 Dataset
- 3,240 records, 16 features
- Source: Kaggle — Manufacturing Defect Dataset
- No missing values or duplicates
- Target: DefectStatus (binary) — imbalanced (~84% defect / 16% no defect)

## 🧹 Data Cleaning & EDA
- Verified no missing values / duplicates
- Checked outliers using IQR method (none found)
- Analyzed correlations between features and target
- Visualized distributions and boxplots by defect status

## 🤖 Model
- Algorithm: Random Forest Classifier (class_weight='balanced')
- Accuracy: 95%
- ROC-AUC: 0.84
- Compared against Logistic Regression baseline (74% accuracy)

Top predictive features:
1. Maintenance Hours
2. Defect Rate
3. Quality Score
4. Production Volume

## 🚀 Streamlit App
An interactive app where you input production/material parameters and get:
- Defect prediction
- Confidence score
- Actionable notes based on key risk factors

### Run locally
\\\bash
pip install -r requirements.txt
streamlit run app.py
\\\

## 🛠️ Tech Stack
Python · Pandas · Scikit-learn · Seaborn/Matplotlib · Streamlit

## 📌 Future Improvements
- Hyperparameter tuning (GridSearchCV)
- Try XGBoost
- Handle imbalance with SMOTE
- Deploy on Streamlit Community Cloud
