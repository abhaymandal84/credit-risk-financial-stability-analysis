import streamlit as st
import joblib, numpy as np

model = joblib.load("credit_risk_model.pkl")

st.title("Credit Risk Predictor")

age = st.slider("Age", 18, 75, 30)
income = st.number_input("Annual Income (₹)", value=500000)
loan_amount = st.number_input("Loan Amount (₹)", value=100000)
credit_score = st.slider("Credit Score", 300, 900, 650)

if st.button("Predict Risk"):
    features = np.array([[age, income, loan_amount, credit_score]])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    
    if pred == 1:
        st.error(f"High Risk — Probability: {prob:.1%}")
    else:
        st.success(f"Low Risk — Probability: {prob:.1%}")