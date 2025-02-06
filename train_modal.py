import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (Make sure to update the dataset path)
df = pd.read_csv("/home/rgukt/Downloads/telecommunications_churn.csv")

# Selecting features (Update column names based on your dataset)
features = ['account_length', 'voice_mail_messages', 'day_mins', 'evening_mins', 
            'night_mins', 'international_mins', 'customer_service_calls']

X = df[features]
y = df['churn']  # Replace 'churn' with the actual target column name

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")

# --- 🔥 Additional Visualization: Confusion Matrix ---
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save the plot
plt.show()

