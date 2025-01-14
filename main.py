import pandas as pd

# Dataset paths
TRUE_PATH = "data/True.csv"  # Path to the true news dataset
FAKE_PATH = "data/Fake.csv"  # Path to the fake news dataset

# Load datasets
try:
    true_data = pd.read_csv(TRUE_PATH)  # Load true news
    fake_data = pd.read_csv(FAKE_PATH)  # Load fake news
    print("Datasets successfully loaded!")

    # Display the first 5 rows of each dataset
    print("\nFirst 5 rows of true news:")
    print(true_data.head())
    print("\nFirst 5 rows of fake news:")
    print(fake_data.head())

    # Check the dimensions of each dataset
    print("\nTrue news shape:", true_data.shape)
    print("Fake news shape:", fake_data.shape)
except FileNotFoundError as e:
    print(f"Dataset not found: {e}")

# Add labels to the datasets
# Label '1' for true news, '0' for fake news
true_data["label"] = 1
fake_data["label"] = 0

# Combine the datasets
data = pd.concat([true_data, fake_data], ignore_index=True)

# Check the dimensions and display the first rows of the combined dataset
print("\nCombined dataset:")
print(data.head())
print("\nCombined dataset shape:", data.shape)

import string

# Remove unnecessary columns (keep only 'text' and 'label')
if 'text' in data.columns:
    data = data[['text', 'label']]
else:
    print("Warning: 'text' column not found. Please check your dataset.")

# Check and clean missing values
print("\nNumber of missing values:")
print(data.isnull().sum())
data = data.dropna()
print("\nMissing values cleaned.")

# Text cleaning function
# - Converts text to lowercase
# - Removes punctuation
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Clean the text data
data['text'] = data['text'].apply(clean_text)

# Display the first 5 rows of the cleaned dataset
print("\nFirst 5 rows of the cleaned dataset:")
print(data.head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Define TF-IDF vectorizer (limit to 5000 words)
tfidf = TfidfVectorizer(max_features=5000)

# Transform the text data into a TF-IDF matrix
X = tfidf.fit_transform(data['text']).toarray()
y = data['label']

print("\nTF-IDF matrix created!")
print(f"Feature matrix shape: {X.shape}")

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDataset split into training and testing sets.")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("\nModel successfully trained!")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.2f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib

# Save the model
MODEL_PATH = "saved_model/logistic_regression_model.pkl"
joblib.dump(model, MODEL_PATH)
print(f"\nModel successfully saved to '{MODEL_PATH}'!")

# Load the saved model and test it
loaded_model = joblib.load(MODEL_PATH)
print("\nSaved model successfully loaded!")

# Make predictions using the loaded model
y_pred_loaded = loaded_model.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"\nAccuracy of the reloaded model: {accuracy_loaded:.2f}")

# Save the TF-IDF vectorizer
TFIDF_PATH = "saved_model/tfidf_vectorizer.pkl"
joblib.dump(tfidf, TFIDF_PATH)
print(f"\nTF-IDF vectorizer successfully saved to '{TFIDF_PATH}'!")
