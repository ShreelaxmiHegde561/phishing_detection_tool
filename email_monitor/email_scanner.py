import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load your dataset (using Nazario.csv as specified)
df = pd.read_csv('C:/Users/shree/OneDrive/Desktop/phishing_detection_tool/data/Nazario.csv')

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Optionally, drop rows with missing values (you can choose a different strategy)
df = df.dropna()

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(df.head())

# Encode categorical features (update this list based on your dataset)
label_encoders = {}
categorical_cols = ['sender', 'receiver', 'date', 'subject', 'body', 'url', 'type', 'text_combined']  # Add any other categorical columns as needed

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure all values are strings for encoding
        label_encoders[col] = le

# Define features and target variable (update 'CLASS_LABEL' with your actual label column name)
X = df.drop(columns=['CLASS_LABEL'])  # Adjust based on your actual label column name
y = df['CLASS_LABEL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
output_dir = 'models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

joblib.dump(model, os.path.join(output_dir, 'phishing_model.pkl'))

print("Model training complete and saved to models/phishing_model.pkl.")
