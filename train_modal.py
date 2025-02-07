import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 Load dataset (Update the path as per your dataset)
df = pd.read_csv("/home/rgukt/Downloads/telecommunications_churn.csv")

# 🔹 Selecting features (Ensure these match your dataset)
features = ['account_length', 'voice_mail_messages', 'day_mins', 'evening_mins', 
            'night_mins', 'international_mins', 'customer_service_calls']

# 🔹 Feature Engineering: Adding "Total Minutes"
df["total_mins"] = df["day_mins"] + df["evening_mins"] + df["night_mins"]
features.append("total_mins")  # Adding new feature

X = df[features]
y = df['churn']  # Ensure this is your actual target column name

# 🔹 Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train Optimized Random Forest Model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 🔹 Predictions
y_pred = model.predict(X_test)

# 🔹 Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")

# 🔹 Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved successfully!")

# 🔹 Feature Importance Visualization (Bar Chart)
feature_importance = model.feature_importances_

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=features, palette="viridis")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Churn Prediction")
plt.savefig("feature_importance.png")
plt.show()

# 🔹 Feature Importance Visualization (Pie Chart)
plt.figure(figsize=(7, 7))
plt.pie(feature_importance, labels=features, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title("Feature Importance (Pie Chart)")
plt.savefig("feature_importance_pie.png")
plt.show()

# 🔹 Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

