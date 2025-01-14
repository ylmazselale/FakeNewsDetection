# Fake News Detection Project

This project is a **Machine Learning** model and API application that classifies news texts as "Real" or "Fake." The following instructions explain how to set up, run, and understand the project.

---

## Features
- **Machine Learning Model:** Trained using the Logistic Regression algorithm.
- **Natural Language Processing (NLP):** Texts are converted into numerical data using the TF-IDF vectorization method.
- **RESTful API:** Developed using Flask to enable predictions for input texts.
- **Results Visualization:** Analyzes prediction results using Matplotlib and Seaborn.

---

## Setup

Follow these steps to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/FakeNewsDetection.git
cd FakeNewsDetection
```

### 2. Create a Virtual Environment
Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Model
To train and test the machine learning model:
```bash
python main.py
```

### 2. Start the Flask API
To start the API:
```bash
python app.py
```
The API will be accessible at `http://127.0.0.1:5000`.

#### Example API Usage
**POST Request:**
```json
{
    "text": "The government announced new policies today."
}
```
**Response:**
```json
{
    "prediction": "Real",
    "confidence": "85.23%"
}
```

### 3. Visualization
To generate and display graphs of the prediction results:
```bash
python visualization.py
```
The graphs will be displayed on the screen and can be saved as files if needed.

---

## Project Structure

```
FakeNewsDetection/
├── app.py                 # Flask API application
├── main.py                # Model training and testing
├── visualization.py       # Visualization of prediction results
├── requirements.txt       # Project dependencies
├── saved_model/           # Saved model and TF-IDF vectorizer
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
├── data/                  # Dataset
│   ├── True.csv
│   └── Fake.csv
└── README.md              # Project documentation



