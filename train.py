import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def run_training():
    """
    Load dataset, preprocess, and train a Logistic Regression model.
    """

    # Ensure file exists (important for CI)
    file_path = 'iris.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found in working directory")

    # Load dataset safely
    df = pd.read_csv(file_path).dropna()

    # Validate required columns
    required_cols = [
        'SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm', 'Species'
    ]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Dataset missing required columns")

    # Features and target
    X = df[['SepalLengthCm', 'SepalWidthCm',
            'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']

    # Encode labels (robust for CI)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )

    # Model (explicit solver for stability)
    model = LogisticRegression(max_iter=300, solver='lbfgs')
    model.fit(X_train, y_train)

    # Optional: simple validation (helps CI pass logic checks)
    acc = model.score(X_test, y_test)
    print(f"Training completed. Accuracy: {acc:.4f}")

if __name__ == '__main__':
    run_training()