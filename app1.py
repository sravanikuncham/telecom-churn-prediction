import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load confusion matrix image
confusion_matrix_img = "confusion_matrix.png"

# Streamlit UI
st.title("📡 Telecom Churn Prediction App 🚀")
st.write("Enter customer details to predict churn & explore model insights.")

# --- Sidebar for Dataset Insights ---
st.sidebar.header("📊 Model Insights")
st.sidebar.image(confusion_matrix_img, caption="Confusion Matrix", use_column_width=True)
st.sidebar.write("This model is trained using a **Decision Tree Classifier**.")
st.sidebar.write("**Accuracy:** 85% (Example, update based on model)")

# Input fields
st.header("🔢 Customer Details")

account_length = st.number_input("📅 Account Length (in days)", min_value=0)
voice_mail_messages = st.number_input("📨 Voice Mail Messages", min_value=0)
day_mins = st.number_input("📞 Day Minutes Used", min_value=0.0)
evening_mins = st.number_input("🌙 Evening Minutes Used", min_value=0.0)
night_mins = st.number_input("🌃 Night Minutes Used", min_value=0.0)
international_mins = st.number_input("🌍 International Minutes Used", min_value=0.0)
customer_service_calls = st.number_input("☎️ Customer Service Calls", min_value=0)

# Predict button
if st.button("🔮 Predict"):
    # Prepare input data
    input_data = np.array([[account_length, voice_mail_messages, day_mins, evening_mins, 
                            night_mins, international_mins, customer_service_calls]])

    # Predict using the model
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.error("⚠️ Customer is **likely to churn**! 😞")
    else:
        st.success("✅ Customer is **NOT likely to churn**! 😊")

# --- 🔥 Additional Visualization: Feature Importance ---
st.subheader("📈 Feature Importance")
feature_importance = model.feature_importances_
features = ['Account Length', 'Voicemail Messages', 'Day Mins', 'Evening Mins', 
            'Night Mins', 'International Mins', 'Customer Service Calls']

plt.figure(figsize=(8, 4))
sns.barplot(x=feature_importance, y=features, palette="viridis")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Churn Prediction")
plt.savefig("feature_importance.png")  # Save the plot
st.image("feature_importance.png", caption="Feature Importance", use_column_width=True)

