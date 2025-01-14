import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_predictions(results):
    """
    Visualizes model prediction results.

    Parameters:
    - results: List[dict] - A list containing prediction results for each news item.

    Example:
    results = [
        {"text": "News 1", "prediction": "Real", "confidence": "85.23%"},
        {"text": "News 2", "prediction": "Fake", "confidence": "92.45%"}
    ]
    """
    if not results or not isinstance(results, list):
        print("Invalid result data provided. Please provide a list.")
        return

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    if 'prediction' not in df.columns or 'confidence' not in df.columns:
        print("The results are missing 'prediction' or 'confidence' columns.")
        return

    # 1. Visualize the distribution of Real and Fake predictions
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='prediction', palette='viridis')
    plt.title('Distribution of Real/Fake News Predictions')
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.show()

    # 2. Visualize the confidence levels of predictions
    try:
        # Convert percentage values to numbers
        df['confidence'] = df['confidence'].str.replace('%', '').astype(float)

        plt.figure(figsize=(8, 6))
        sns.histplot(df['confidence'], kde=True, bins=10, color='blue')
        plt.title('Distribution of Prediction Confidence Levels')
        plt.xlabel('Confidence Level (%)')
        plt.ylabel('Frequency')
        plt.show()
    except Exception as e:
        print(f"An error occurred while visualizing confidence levels: {e}")
