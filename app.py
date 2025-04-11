import train_model  # Assuming this is your custom module for training
import joblib  # For loading the trained model and vectorizer
import pandas as pd  # For handling data
import os

# Load the trained model and vectorizer
model = joblib.load('models/phishing_model.pkl')  # Path to the saved model
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')  # Path to the saved vectorizer

# Initialize a log to keep track of predictions
prediction_log = []

def get_email_details():
    """Collect email content and optional URL, along with metadata."""
    print("\n--- Phishing Detection Tool ---")
    
    headers = input("\nPlease enter the email headers (or type 'exit' to quit):\n").strip()
    if headers.lower() == 'exit':
        return None, None, None, None, None

    sender_ip = input("\nPlease enter the sender's IP address:\n").strip()
    
    attachments = input("\nPlease enter the attachment file paths (comma-separated) or press Enter to skip:\n").strip()
    attachments_list = [a.strip() for a in attachments.split(',')] if attachments else []

    email_content = input("\nPlease paste the email content (or type 'exit' to quit):\n").strip()
    if email_content.lower() == 'exit':
        return None, None, None, None, None

    while not email_content:
        print("Email content cannot be empty. Please try again.")
        email_content = input("Please paste the email content:\n").strip()

    url_content = input("\n(Optional) If the email contains a suspicious URL, please paste it here (or press Enter to skip):\n").strip()
    if not url_content:
        print("No URL provided. Proceeding without URL.")
        url_content = 'null'

    return headers, sender_ip, email_content, url_content, attachments_list

def preprocess_data(headers, sender_ip, email_content, url_content, attachments):
    """Preprocess the input data for model prediction."""
    full_input = f"{headers} {sender_ip} {' '.join(attachments)} {email_content} {url_content}"
    # Optionally print for debugging
    # print("\n[DEBUG] Full input to model:", full_input)
    return vectorizer.transform([full_input])

def log_prediction(email_content, prediction, confidence):
    """Log each prediction for user reference."""
    log_entry = {
        'email_content': email_content[:100],  # Only store first 100 characters
        'prediction': "Safe" if prediction == 0 else "Phishing",
        'confidence': f"{confidence * 100:.2f}%"
    }
    prediction_log.append(log_entry)

def display_log():
    """Display the log of predictions made during the session."""
    if prediction_log:
        print("\n--- Prediction Log ---")
        for i, entry in enumerate(prediction_log, 1):
            print(f"\nEmail {i}:")
            print(f"Preview: {entry['email_content']}")
            print(f"Prediction: {entry['prediction']}")
            print(f"Confidence: {entry['confidence']}")
    else:
        print("\nNo predictions made yet.")

def main():
    """Main function to run the phishing detection tool."""
    print("Welcome to the Phishing Detection Tool!")

    while True:
        headers, sender_ip, email_content, url_content, attachments = get_email_details()

        if headers is None and sender_ip is None and email_content is None and url_content is None and not attachments:
            print("\nThank you for using the tool! Here's the log of your session:")
            display_log()
            print("Goodbye!")
            break

        input_data = preprocess_data(headers, sender_ip, email_content, url_content, attachments)

        prediction = model.predict(input_data)
        confidence_score = model.predict_proba(input_data)[0][prediction[0]]

        result = "Safe" if prediction[0] == 0 else "Warning: This email might be a Phishing Attempt!"
        print("\n--- Prediction Result ---")
        print(f"{result} (Confidence: {confidence_score * 100:.2f}%)")
        print(f"Sender IP: {sender_ip}")
        print(f"Attachments: {', '.join(attachments) if attachments else 'No attachments'}")

        log_prediction(email_content, prediction[0], confidence_score)

        another = input("\nWould you like to test another email? (yes/no): ").strip().lower()
        if another != 'yes':
            print("\nThank you for using the Phishing Detection Tool! Here's the log of your session:")
            display_log()
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
