from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Create the Flask application
app = Flask(__name__)

# Load the saved model and TF-IDF vectorizer
MODEL_PATH = "saved_model/logistic_regression_model.pkl"
TFIDF_PATH = "saved_model/tfidf_vectorizer.pkl"
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

@app.route('/')
def home():
    """
    Homepage endpoint. Used to check if the API is running.
    """
    return "Fake News Detection API is running! Use /predict or /predict_batch endpoints for POST requests."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Analyzes a single text and predicts whether it is fake or real.
    Expected input: 'text' parameter in JSON format.
    """
    # Get JSON data
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide the 'text' parameter!"}), 400

    # Extract and vectorize the text
    text = [data['text']]
    vectorized_text = tfidf.transform(text).toarray()

    # Make prediction with the model
    prediction = model.predict(vectorized_text)[0]
    confidence = model.predict_proba(vectorized_text).max() * 100

    # Prepare the results
    result = {
        "prediction": "Real" if prediction == 1 else "Fake",
        "confidence": f"{confidence:.2f}%"
    }

    return jsonify(result)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Analyzes multiple texts and predicts whether each is fake or real.
    Expected input: 'texts' parameter (list) in JSON format.
    """
    # Get JSON data
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({"error": "Please provide the 'texts' parameter with a list of texts!"}), 400

    texts = data['texts']
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list!"}), 400

    # Vectorize the texts
    vectorized_texts = tfidf.transform(texts).toarray()

    # Make predictions and prepare the results
    predictions = model.predict(vectorized_texts)
    confidences = model.predict_proba(vectorized_texts).max(axis=1)

    results = [
        {
            "text": text,
            "prediction": "Real" if pred == 1 else "Fake",
            "confidence": f"{conf * 100:.2f}%"
        }
        for text, pred, conf in zip(texts, predictions, confidences)
    ]

    return jsonify(results)

if __name__ == '__main__':
    """
    Starts the Flask application in development mode.
    """
    app.run(debug=True)
