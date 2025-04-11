import pandas as pd # type: ignore
import os

data_directory = 'data/'

files_to_combine = [
    'CEAS_08.csv',
    'Enron.csv',
    'Ling.csv',
    'malicious_phish.csv',
    'Nazario.csv',
    'Nigerian_Fraud.csv',
    'phishing_email.csv',
    'Phishing_Legitimate_full.csv',
    'SpamAssasin.csv'
]

dataframes = []
missing_files = []

for file in files_to_combine:
    file_path = os.path.join(data_directory, file)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded: {file} | Shape: {df.shape}")
            dataframes.append(df)
        except Exception as e:
            print(f"⚠️ Failed to read {file}: {e}")
    else:
        print(f"❌ File not found: {file}")
        missing_files.append(file)

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(os.path.join(data_directory, 'combined_dataset.csv'), index=False)
    print(f"\n✅ Combined dataset saved as 'combined_dataset.csv'. Total rows: {combined_df.shape[0]}")
else:
    print("\n❌ No datasets were loaded. Check missing files or file errors above.")
