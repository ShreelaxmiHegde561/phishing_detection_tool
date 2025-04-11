import re
import pandas as pd

def extract_email_features(email_content):
    # Dummy feature extraction
    features = []
    features.append(len(email_content))  # Length of the email
    features.append(1 if "http" in email_content else 0)  # Check for links
    features.append(1 if re.search(r'\bfree\b', email_content, re.I) else 0)  # Check for 'free' keyword
    return pd.DataFrame([features])
