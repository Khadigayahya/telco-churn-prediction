# Telecom Customer Churn Prediction

A machine learning web app that predicts whether a telecom customer is likely to **churn** (leave the company), built with **scikit-learn** and deployed with **Streamlit**.

**Live app:** https://telco-churn-prediction-lsm9fmgu2zimwgerznetw9.streamlit.app

---

## Problem
Acquiring a new customer costs much more than keeping an existing one. This project identifies customers at risk of leaving, so the company can act before they churn.

- **Type:** Supervised learning, binary classification
- **Target:** `Churn` (Yes / No)
- **Main metric:** Recall and F1-score for the churn class

## Dataset
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle: 7,043 customers and 21 columns (demographics, services, account and billing information).

## Workflow
1. **Data definition:** structure, data types, data dictionary
2. **Data cleaning:** fixed 11 hidden blank values in `TotalCharges`, wrong data types, removed 22 duplicates
3. **EDA & visualization:** univariate, bivariate and correlation analysis
4. **Outlier detection:** IQR and Z-score (no outliers found)
5. **Feature engineering:** simplified redundant categories, created `NumServices`, One-Hot Encoding and scaling in a `ColumnTransformer`
6. **Modeling:** 7 models compared with and without **SMOTE**
7. **Hyperparameter tuning:** GridSearchCV / RandomizedSearchCV with 5-fold stratified CV
8. **Evaluation:** confusion matrix, ROC and PR curves, overfitting check, feature importance, threshold analysis

## Final Model: Gradient Boosting (tuned) + SMOTE

| Metric | Score |
|---|---|
| Recall | 0.761 |
| F1-score | 0.628 |
| ROC-AUC | 0.840 |
| Accuracy | 0.761 |

**Top churn drivers:** month-to-month contract, low tenure, fiber optic internet, electronic check payment.

## App Features
- **Single customer prediction:** churn probability gauge, risk level and retention recommendations
- **Batch prediction:** upload a CSV, get predictions for all customers and download the results
- **Model insights:** performance metrics and feature importance

## Project Structure
```
├── app.py                  # Streamlit application
├── churn_model.pkl         # Trained pipeline (preprocessing + SMOTE + model)
├── model_metadata.json     # Input options, metrics and feature importance
├── requirements.txt
├── notebook/
│   └── Telco_Churn_Analysis.ipynb   # Full analysis (Google Colab)
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## ▶️ Run Locally
Requires **Python 3.13** (the model was saved with scikit-learn 1.6.1).

```bash
py -3.13 -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Author
Khadiga Yahya
