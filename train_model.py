import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import joblib # type: ignore

# Load the dataset
df = pd.read_csv("data/combined_dataset.csv")
print("\n✅ Dataset loaded successfully.")
print(f"Shape: {df.shape}")

# Try to infer the correct 'text' and 'label' columns
possible_text_columns = ['email', 'text', 'message', 'body', 'content']
possible_label_columns = ['label', 'phishing', 'is_phishing', 'target']

text_col = next((col for col in df.columns if col.lower() in possible_text_columns), None)
label_col = next((col for col in df.columns if col.lower() in possible_label_columns), None)

if text_col is None or label_col is None:
    print("\n❌ Could not automatically detect 'email' or 'label' column.")
    print(f"Detected columns: {df.columns.tolist()}")
    exit(1)

print(f"\n🧠 Using '{text_col}' as text column and '{label_col}' as label column.")

# Drop missing values
df = df[[text_col, label_col]].dropna()

# Show label distribution
print("\nLabel distribution:")
print(df[label_col].value_counts())

# Features and labels
X = df[text_col].astype(str)
y = df[label_col]

# Ensure labels are binary (0 or 1)
y = y.map({0: 0, 1: 1}).fillna(0).astype(int)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n🧪 Evaluation Results:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\n✅ Model and vectorizer saved successfully.")
