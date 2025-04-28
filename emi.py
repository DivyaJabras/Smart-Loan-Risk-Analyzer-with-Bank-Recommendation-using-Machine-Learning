import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('loan_model.pkl')

# Bank Offers (Hardcoded)
bank_offers = pd.DataFrame({
    "Bank Name": ["HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra"],
    "Min Risk Level": ["Low", "Medium", "Low", "High", "Medium"],
    "ROI (%)": [9.5, 11.0, 10.2, 13.5, 10.8],
    "Loan Term (months)": [60, 48, 72, 36, 60]
})

def calculate_emi(principal, annual_roi, months):
    monthly_rate = (annual_roi / 12) / 100
    emi = (principal * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
    return round(emi, 2)

# Title
st.title("🏦 FinTech Loan Predictor & Bank Recommender")

# Tabs via Sidebar
page = st.sidebar.radio("Choose Feature", ["📊 Loan Approval Prediction", "🏦 Bank Suggestion"])

# Common Inputs
st.sidebar.header("Applicant Info")
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
married = st.sidebar.selectbox("Married", ['Yes', 'No'])
dependents = st.sidebar.selectbox("Dependents", ['0', '1', '2', '3+'])
education = st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.sidebar.selectbox("Self Employed", ['Yes', 'No'])
applicant_income = st.sidebar.number_input("Applicant Income", 0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", 0)
loan_amount = st.sidebar.number_input("Loan Amount (in ₹)", 100000, step=10000)
loan_amount_thousands = loan_amount / 1000
loan_amount_term = st.sidebar.number_input("Loan Term (in days)", 0)
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# Input mapping
input_data = pd.DataFrame({
    'Gender': [1 if gender == 'Male' else 0],
    'Married': [1 if married == 'Yes' else 0],
    'Dependents': [0 if dependents == '0' else 1 if dependents == '1' else 2 if dependents == '2' else 3],
    'Education': [0 if education == 'Graduate' else 1],
    'Self_Employed': [1 if self_employed == 'Yes' else 0],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount_thousands],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [0 if property_area == 'Rural' else 1 if property_area == 'Semiurban' else 2]
})

# ---------------- TAB 1 ----------------
if page == "📊 Loan Approval Prediction":
    st.subheader("📊 Prediction Result")
    if st.button("🔍 Predict Loan Approval"):
        approval = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        risk_score = round(prob * 100, 2)

        if risk_score < 50:
            risk_level = '🔴 High Risk'
        elif risk_score < 80:
            risk_level = '🟠 Medium Risk'
        else:
            risk_level = '🟢 Low Risk'

        st.success(f"**Loan Approved:** {'✅ Yes' if approval == 1 else '❌ No'}")
        st.info(f"**Approval Probability:** {round(prob, 2)}")
        st.info(f"**Risk Score:** {risk_score}")
        st.info(f"**Risk Level:** {risk_level}")

# ---------------- TAB 2 ----------------
elif page == "🏦 Bank Suggestion":
    st.subheader("🏦 Bank Recommendation Based on Risk")
    if st.button("📌 Suggest Best Banks"):
        # Get probability & risk
        prob = model.predict_proba(input_data)[0][1]
        risk_score = round(prob * 100, 2)

        if risk_score < 50:
            user_risk = "High"
        elif risk_score < 80:
            user_risk = "Medium"
        else:
            user_risk = "Low"

        st.write(f"🧠 **Predicted Risk Score:** {risk_score}")
        st.write(f"📊 **Inferred Risk Level:** {user_risk}")

        # Filter suitable banks
        suitable_banks = bank_offers[bank_offers["Min Risk Level"] <= user_risk]

        if not suitable_banks.empty:
            result = suitable_banks.copy()
            result["EMI (₹)"] = result.apply(lambda row: calculate_emi(loan_amount, row["ROI (%)"], row["Loan Term (months)"]), axis=1)
            st.dataframe(result[["Bank Name", "ROI (%)", "Loan Term (months)", "EMI (₹)"]])
        else:
            st.warning("⚠️ No suitable bank found for your risk profile.")
