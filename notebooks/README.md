# Employee Attrition Prediction

A Machine Learning project that predicts whether an employee is likely to leave an organization based on selected employee and work-related attributes.

## Project Overview

Employee attrition refers to employees leaving an organization.

This project uses the IBM HR Analytics Employee Attrition & Performance dataset to train a Machine Learning classification model.

The project performs:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature selection
- Model training
- Employee attrition prediction
- Streamlit web application deployment

## Dataset

Dataset: IBM HR Analytics Employee Attrition & Performance

The dataset contains 1470 employee records and 35 attributes.

The target variable is:

- `Attrition = Yes` → Employee left
- `Attrition = No` → Employee stayed

## Selected Features

The final model uses 12 selected features:

1. OverTime
2. JobRole
3. BusinessTravel
4. MaritalStatus
5. YearsAtCompany
6. TotalWorkingYears
7. YearsWithCurrManager
8. Age
9. MonthlyIncome
10. JobLevel
11. JobSatisfaction
12. YearsInCurrentRole

## Machine Learning Model

The final model is:

**Logistic Regression with class balancing**

Preprocessing includes:
- One-Hot Encoding for categorical features
- Standard Scaling for numerical features
- Train-test split with stratification

## Model Performance

The final model achieved approximately:

- Accuracy: 75%
- Attrition (Yes) Recall: 66%
- Attrition (Yes) F1-score: 46%

The model is intended as a prototype based on the characteristics of the selected dataset.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Git & GitHub

## Project Structure

```text
Employee-Attrition-Prediction/
│
├── dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── models/
│   └── employee_attrition_model.pkl
│
├── notebooks/
│   ├── eda.py
│   └── feature_selection.py
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md