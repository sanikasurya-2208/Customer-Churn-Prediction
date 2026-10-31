
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load(
    "Trained Model/customer_churn_model.pkl"
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict the possibility of churn.")

st.divider()

st.subheader("Customer Information")

# Customer details
gender = st.selectbox("Gender", ["Female", "Male"])

senior_citizen = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input(
    "Tenure (months)", min_value=0, max_value=72, value=12
)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])

multiple_lines = st.selectbox(
    "Multiple Lines", ["No", "Yes", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing", ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)

total_charges = st.number_input(
    "Total Charges", min_value=0.0, value=840.0
)


st.divider()


# Prediction button
if st.button("Predict Customer Churn", type="primary"):

    # Prepare customer data
    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Predict churn
    prediction = model.predict(customer_data)

    # Calculate probability
    probability = model.predict_proba(customer_data)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    if prediction[0] == 1:
        st.error("Customer is likely to leave (Churn: Yes)")
    else:
        st.success("Customer is likely to stay (Churn: No)")

    st.caption(
        "Probability represents the model's estimated likelihood of customer churn."
    )
