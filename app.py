import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Set page configuration
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# Title and description
st.title("💳 Credit Card Default Prediction")
st.markdown("""
This app predicts the probability of credit card default based on various customer attributes.
Please input the required information below to get a prediction.
""")

def load_data():
    # Load the data and prepare the model
    data = pd.read_csv("creditCardFraud_28011964_120214.csv")
    data.rename(columns={"PAY_0": "PAY_1"}, inplace=True)
    return data

def train_model(data):
    # Prepare the data
    X = data.drop("default payment next month", axis=1)
    y = data["default payment next month"]
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def main():
    try:
        # Load data and train model
        data = load_data()
        model, scaler = train_model(data)
        
        # Create columns for input fields
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Information")
            limit_bal = st.number_input("Credit Limit Balance", min_value=0, value=50000)
            sex = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
            education = st.selectbox("Education", options=[1, 2, 3, 4], 
                                   format_func=lambda x: {1: "Graduate School", 2: "University", 
                                                        3: "High School", 4: "Others"}[x])
            marriage = st.selectbox("Marital Status", options=[1, 2, 3], 
                                  format_func=lambda x: {1: "Married", 2: "Single", 
                                                       3: "Others"}[x])
            age = st.number_input("Age", min_value=18, max_value=100, value=25)

        with col2:
            st.subheader("Payment Status History")
            pay_1 = st.selectbox("Payment Status (Month 1)", 
                               options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            pay_2 = st.selectbox("Payment Status (Month 2)", 
                               options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            pay_3 = st.selectbox("Payment Status (Month 3)", 
                               options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            pay_4 = st.selectbox("Payment Status (Month 4)", 
                               options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            pay_5 = st.selectbox("Payment Status (Month 5)", 
                               options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            pay_6 = st.selectbox("Payment Status (Month 6)", 
                               options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        with col3:
            st.subheader("Bill and Payment Amounts")
            # Bill Amounts
            bill_amt1 = st.number_input("Bill Amount (Month 1)", value=0)
            bill_amt2 = st.number_input("Bill Amount (Month 2)", value=0)
            bill_amt3 = st.number_input("Bill Amount (Month 3)", value=0)
            bill_amt4 = st.number_input("Bill Amount (Month 4)", value=0)
            bill_amt5 = st.number_input("Bill Amount (Month 5)", value=0)
            bill_amt6 = st.number_input("Bill Amount (Month 6)", value=0)
            
            # Payment Amounts
            pay_amt1 = st.number_input("Payment Amount (Month 1)", value=0)
            pay_amt2 = st.number_input("Payment Amount (Month 2)", value=0)
            pay_amt3 = st.number_input("Payment Amount (Month 3)", value=0)
            pay_amt4 = st.number_input("Payment Amount (Month 4)", value=0)
            pay_amt5 = st.number_input("Payment Amount (Month 5)", value=0)
            pay_amt6 = st.number_input("Payment Amount (Month 6)", value=0)

        # Create input array for prediction
        input_data = pd.DataFrame([[limit_bal, sex, education, marriage, age, 
                                  pay_1, pay_2, pay_3, pay_4, pay_5, pay_6,
                                  bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
                                  pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]], 
                                columns=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                                       'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                                       'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                                       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])

        # Add predict button
        if st.button("Predict Default Probability"):
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Default Probability", f"{prediction[1]:.2%}")
            
            with col2:
                st.metric("Non-Default Probability", f"{prediction[0]:.2%}")
            
            # Display risk level
            risk_level = "High Risk 🔴" if prediction[1] > 0.7 else "Medium Risk 🟡" if prediction[1] > 0.3 else "Low Risk 🟢"
            st.markdown(f"### Risk Assessment: {risk_level}")
            
            # Additional insights
            st.markdown("### Key Insights")
            insights = []
            if pay_1 > 0:
                insights.append("⚠️ Recent payment delays may increase default risk")
            if limit_bal < 50000:
                insights.append("📊 Lower credit limit might indicate limited credit history")
            if bill_amt1 > limit_bal:
                insights.append("💰 Current bill amount exceeds credit limit")
            if any([pay_1, pay_2, pay_3, pay_4, pay_5, pay_6]) > 2:
                insights.append("🚫 Multiple payment delays in history indicate higher risk")
            
            for insight in insights:
                st.markdown(f"- {insight}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.markdown("Please make sure all required files are available and input values are valid.")

if __name__ == "__main__":
    main()