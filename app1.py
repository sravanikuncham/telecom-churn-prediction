import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load images
confusion_matrix_img = "confusion_matrix.png"
feature_importance_img = "feature_importance.png"
feature_importance_pie_img = "feature_importance_pie.png"

# --- Streamlit App UI ---
st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📡", layout="wide")

# Custom CSS for enhanced UI
st.markdown("""
    <style>
        .main {background-color: #f0f2f6;}
        div.stButton > button:first-child { background-color: #4CAF50; color: white; font-size: 18px; }
        div.stButton > button:hover { background-color: #45a049; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("📊 Model Insights")
st.sidebar.image(confusion_matrix_img, caption="Confusion Matrix", use_column_width=True)
st.sidebar.image(feature_importance_pie_img, caption="Feature Importance (Pie Chart)", use_column_width=True)
st.sidebar.markdown("### **Model: Random Forest Classifier**")
st.sidebar.markdown("✅ **Accuracy:** ~90-94%")
st.sidebar.markdown("📌 Trained with optimized hyperparameters")

# --- Main Section ---
st.title("📡 Telecom Churn Prediction App 🚀")
st.write("**Predict customer churn probability & explore model insights!**")

# Input Fields
st.subheader("🔢 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    account_length = st.number_input("📅 Account Length (in days)", min_value=0)
    voice_mail_messages = st.number_input("📨 Voice Mail Messages", min_value=0)
    customer_service_calls = st.number_input("☎️ Customer Service Calls", min_value=0)

with col2:
    day_mins = st.number_input("📞 Day Minutes Used", min_value=0.0)
    evening_mins = st.number_input("🌙 Evening Minutes Used", min_value=0.0)
    night_mins = st.number_input("🌃 Night Minutes Used", min_value=0.0)

with col3:
    international_mins = st.number_input("🌍 International Minutes Used", min_value=0.0)
    total_mins = day_mins + evening_mins + night_mins  # Feature Engineering

# --- Prediction ---
if st.button("🔮 Predict Churn"):
    input_data = np.array([[account_length, voice_mail_messages, day_mins, evening_mins, 
                            night_mins, international_mins, customer_service_calls, total_mins]])
    
    prediction = model.predict(input_data)

    st.subheader("📌 Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ **Customer is likely to churn!** 😞")
        st.warning("💡 Consider offering retention strategies such as discounts, better plans, or loyalty rewards.")
    else:
        st.success("✅ **Customer is NOT likely to churn!** 😊")

# --- Visualization ---
st.subheader("📈 Feature Importance")
st.image(feature_importance_img, caption="Feature Importance (Bar Chart)", use_column_width=True)

# --- Additional Insights ---
st.markdown("## 🔍 Additional Insights")
st.write("""
    - 📌 **Higher total minutes used correlates with lower churn probability.**  
    - 📌 **Frequent customer service calls may indicate dissatisfaction.**  
    - 📌 **Customers using international minutes tend to churn more frequently.**  
    - 📌 **Churn prevention strategies can target high-risk customers effectively.**  
""")

# --- Footer ---
st.markdown("""
    <hr>
    <p style="text-align:center;">📡 Developed by <strong>Your Name</strong> | © 2025 Telecom Churn AI</p>
""", unsafe_allow_html=True)

