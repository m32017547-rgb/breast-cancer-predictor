# 🩺 Breast Cancer Diagnosis Predictor

An end-to-end machine learning project: data → model comparison → interactive web app.

> ⚠️ Educational project only. Not a medical tool.

## What it does
- Predicts whether a tumor is **benign** or **malignant** from 10 cell-nucleus measurements
- Compares 4 models (Logistic Regression, SVM, Random Forest, Gradient Boosting) with 5-fold cross-validation
- Shows metrics, confusion matrix, feature importance, and an interactive data explorer

## Tech stack
Python · scikit-learn · pandas · Plotly · Streamlit

## Results
Best model (chosen by cross-validation): **SVM (RBF)**, about 94% CV accuracy and 0.98 ROC-AUC on the held-out test set.
Full table inside the app (Model performance tab).

## Run locally
```bash
pip install -r requirements.txt
python train_model.py      # trains models, creates model.joblib + metrics.json
streamlit run app.py
```

## Project structure
```
app.py             # Streamlit web app
train_model.py     # training + model comparison
model.joblib       # trained pipeline (scaler + model)
metrics.json       # evaluation results used by the app
requirements.txt
```

## Dataset
Breast Cancer Wisconsin (Diagnostic), 569 samples, included in scikit-learn.

## Author
Mahmoud Emad · AI & Data Science, Al-Zagazig National University
