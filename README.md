# 🏦 Credit Card Default Prediction Using Behavior Score Modeling

## 📘 Project Overview
This project focuses on building a robust classification model to predict whether a credit card customer is likely to default in the next billing cycle. Leveraging real-world anonymized data from over 30,000 customers, the goal is to enable risk-based decision-making by identifying high-risk customers in advance.

By integrating behavioral variables such as payment delays, billing patterns, and credit utilization, the model not only predicts defaults but also provides interpretable financial insights to support credit exposure management and early risk intervention.

---

## 🎯 Objectives
- Build a binary classification model to predict `next_month_default` (1 = Default, 0 = No Default).
- Handle class imbalance using methods like **SMOTE**, **class weighting**, or **undersampling**.
- Perform **EDA + Financial Analysis** to understand behavioral trends.
- Engineer predictive features (e.g., `credit utilization`, `delinquency streaks`, `repayment ratio`).
- Compare and evaluate multiple models:  
  - Logistic Regression  
  - Decision Tree  
  - XGBoost, LightGBM
- Optimize classification threshold aligned with real-world credit risk trade-offs.
- Generate production-ready predictions for a validation dataset.

---

## 📦 Deliverables
- ✅ **Clean Jupyter Notebook** with code for preprocessing, analysis, modeling, and prediction.
- 📊 **EDA + Financial Insight Report** (integrated or separate PDF).
- 📁 **Prediction File**: `submission_<YourEnrollmentNumber>.csv` (Customer ID + Prediction).
- 📈 **Model Performance Report** with metrics:  
  - Accuracy, F1-score, F2-score, Precision, Recall, ROC-AUC  
  - Classification threshold tuning

---

## 🔍 Dataset Details

| Dataset        | Description |
|----------------|-------------|
| **Training Set** | ~25,000 records with labeled `next_month_default` |
| **Validation Set** | ~5,000 records (no labels) for prediction submission |

Key features include:
- Demographics: Age, Sex, Education, Marital Status
- Credit Details: `LIMIT_BAL`, Bill & Payment Amounts (over 6 months)
- Payment Status: `pay_0` to `pay_6`
- Engineered: `AVG_Bill_amt`, `PAY_TO_BILL_ratio`

---

## 🛠 Tools & Libraries
- **Python**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **ML Libraries**: `scikit-learn`, `imbalanced-learn`, `xgboost`, `lightgbm`
- **Interpretability (optional)**: `SHAP`, `LIME`

---

