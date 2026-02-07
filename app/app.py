import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and columns
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_columns = joblib.load("models/model_columns.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details:")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
techsupport = st.selectbox("Tech Support", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", 0, 72)
monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)

if st.button("Predict"):

    # Create dataframe with all columns
    input_data = pd.DataFrame(columns=model_columns)

    # Fill numeric values
    input_data.loc[0, 'tenure'] = tenure
    input_data.loc[0, 'MonthlyCharges'] = monthly
    input_data.loc[0, 'TotalCharges'] = total

    # Fill categorical encoded values
    if gender == "Male":
        input_data.loc[0, 'gender_Male'] = 1

    if partner == "Yes":
        input_data.loc[0, 'Partner_Yes'] = 1

    if dependents == "Yes":
        input_data.loc[0, 'Dependents_Yes'] = 1

    if contract == "One year":
        input_data.loc[0, 'Contract_One year'] = 1
    elif contract == "Two year":
        input_data.loc[0, 'Contract_Two year'] = 1

    if internet == "Fiber optic":
        input_data.loc[0, 'InternetService_Fiber optic'] = 1
    elif internet == "No":
        input_data.loc[0, 'InternetService_No'] = 1

    if techsupport == "Yes":
        input_data.loc[0, 'TechSupport_Yes'] = 1

    # Fill remaining columns with 0
    input_data.fillna(0, inplace=True)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")

        st.success("Customer is likely to Stay")

