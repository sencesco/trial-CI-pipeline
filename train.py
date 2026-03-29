import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os


def run_training():
    """
    A simple function to load data and train a model.
    """
    # Load the dataset
    # Get the directory where the current script lives
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'Iris.csv')
    
    df = pd.read_csv(file_path)
    

    # Define features and target
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize and train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    print("Training completed successfully.")

if __name__ == '__main__':
    run_training()
