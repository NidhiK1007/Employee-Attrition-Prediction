import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 2rem;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 17px;
        margin-bottom: 30px;
    }

    /* Section headings */
    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Information box */
    .info-box {
        padding: 14px 18px;
        border-radius: 10px;
        background-color: #f5f7fa;
        border: 1px solid #e1e5ea;
        margin-bottom: 20px;
        color: #555;
        font-size: 14px;
    }

    /* Prediction card */
    .prediction-card {
        padding: 25px;
        border-radius: 14px;
        background-color: #f5f7fa;
        border: 1px solid #e1e5ea;
        text-align: center;
        margin-top: 25px;
    }

    .prediction-title {
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .probability {
        font-size: 18px;
        margin-top: 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(
    "models/employee_attrition_model.pkl"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">👥 Employee Attrition Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict whether an employee is likely to leave the organization'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-box">'
    'Enter the employee information below. '
    'The model uses 12 selected features to generate the prediction.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# EMPLOYEE PROFILE
# --------------------------------------------------

st.markdown(
    '<div class="section-title">👤 Employee Profile</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=0,
        step=1,
        value=30
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Representative",
            "Laboratory Technician",
            "Human Resources",
            "Sales Executive",
            "Research Scientist",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Research Director"
        ]
    )

with col2:

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5]
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        step=100,
        value=5000
    )


# --------------------------------------------------
# WORK INFORMATION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">💼 Work Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    overtime = st.selectbox(
        "Overtime",
        [
            "Yes",
            "No"
        ]
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        step=1,
        value=5
    )

with col2:

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        step=1,
        value=8
    )

    years_in_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        step=1,
        value=3
    )

    years_with_curr_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        step=1,
        value=3
    )


# --------------------------------------------------
# NOTE
# --------------------------------------------------

st.caption(
    "For best results, enter values within the ranges represented "
    "in the training dataset."
)


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔍 Predict Attrition",
    use_container_width=True
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    input_data = pd.DataFrame({
        "OverTime": [overtime],
        "JobRole": [job_role],
        "BusinessTravel": [business_travel],
        "MaritalStatus": [marital_status],
        "YearsAtCompany": [years_at_company],
        "TotalWorkingYears": [total_working_years],
        "YearsWithCurrManager": [years_with_curr_manager],
        "Age": [age],
        "MonthlyIncome": [monthly_income],
        "JobLevel": [job_level],
        "JobSatisfaction": [job_satisfaction],
        "YearsInCurrentRole": [years_in_current_role]
    })


    # Make prediction
    prediction = model.predict(input_data)[0]


    # Get probabilities
    probabilities = model.predict_proba(input_data)[0]

    class_names = model.classes_

    probability_dict = dict(
        zip(class_names, probabilities)
    )


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == "Yes":

        st.error(
            "⚠️ The model predicts that the employee is likely to leave."
        )

    else:

        st.success(
            "✅ The model predicts that the employee is unlikely to leave."
        )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Leaving Probability",
            f"{probability_dict['Yes'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Staying Probability",
            f"{probability_dict['No'] * 100:.2f}%"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    '<div class="footer">'
    'Employee Attrition Prediction • Machine Learning Mini Project'
    '</div>',
    unsafe_allow_html=True
)