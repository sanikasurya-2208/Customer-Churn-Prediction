
import streamlit as st
import pandas as pd
import joblib
import mysql.connector
import uuid

# Load trained model
model = joblib.load("Trained Model/customer_churn_model.pkl")


# MySQL database connection
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=st.secrets["mysql"]["password"],
        database="customer_churn_db"
    )


# Save customer and prediction in MySQL
def save_prediction(customer_data, prediction, probability, result):

    customer_id = "APP" + uuid.uuid4().hex[:12]

    connection = get_connection()
    cursor = connection.cursor()

    try:
        customer = customer_data.iloc[0]

        # Save customer details
        customer_query = """
        INSERT INTO Customers (
            customerID, gender, SeniorCitizen, Partner,
            Dependents, tenure, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup,
            DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, Contract, PaperlessBilling,
            PaymentMethod, MonthlyCharges, TotalCharges, Churn
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        customer_values = (
            customer_id,
            customer["gender"],
            int(customer["SeniorCitizen"]),
            customer["Partner"],
            customer["Dependents"],
            int(customer["tenure"]),
            customer["PhoneService"],
            customer["MultipleLines"],
            customer["InternetService"],
            customer["OnlineSecurity"],
            customer["OnlineBackup"],
            customer["DeviceProtection"],
            customer["TechSupport"],
            customer["StreamingTV"],
            customer["StreamingMovies"],
            customer["Contract"],
            customer["PaperlessBilling"],
            customer["PaymentMethod"],
            float(customer["MonthlyCharges"]),
            float(customer["TotalCharges"]),
            result
        )

        cursor.execute(customer_query, customer_values)

        # Save prediction details
        prediction_query = """
        INSERT INTO Predictions (
            customerID,
            predicted_churn,
            churn_probability,
            model_name
        )
        VALUES (%s, %s, %s, %s)
        """

        cursor.execute(
            prediction_query,
            (
                customer_id,
                result,
                float(probability),
                "Tuned Random Forest Classifier"
            )
        )

        connection.commit()

        return customer_id

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("Customer Churn")
st.sidebar.write("Machine Learning Project")
st.sidebar.divider()

st.sidebar.info(
    "This application predicts whether a customer "
    "is likely to leave a service."
)

# Main heading
st.title("📊 Customer Churn Prediction")
st.write("Analyze customer information and predict churn.")

st.divider()

# Customer information
st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col2:
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

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
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

# Prediction section
st.header("Churn Prediction")

if st.button("Predict Customer Churn", type="primary"):

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

    prediction = model.predict(customer_data)
    probability = model.predict_proba(customer_data)[0][1]

    result = "Yes" if prediction[0] == 1 else "No"

    st.subheader("Prediction Result")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with metric_col2:
        display_result = (
            "Likely to Leave" if result == "Yes"
            else "Likely to Stay"
        )
        st.metric("Prediction", display_result)

    st.progress(float(probability))

    if result == "Yes":
        st.error("Customer is likely to leave the service.")
    else:
        st.success("Customer is likely to stay with the service.")

    st.caption(
        "Probability is the model's estimated likelihood of churn."
    )

    # Save prediction to MySQL
    try:
        customer_id = save_prediction(
            customer_data,
            prediction,
            probability,
            result
        )

        st.success("Prediction saved successfully in MySQL!")
        st.write("Customer ID:", customer_id)

    except Exception as error:
        st.error(f"Prediction generated, but database saving failed: {error}")

st.divider()

# Model performance
st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "75.09%")
col2.metric("Precision", "52.14%")
col3.metric("Recall", "74.87%")
col4.metric("F1 Score", "61.47%")

st.caption("Model: Tuned Random Forest Classifier")

st.divider()
st.caption("Customer Churn Prediction | Machine Learning Project")
