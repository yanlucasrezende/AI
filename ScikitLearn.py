import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os


def load_and_preprocess(csv_path, test_size=0.2, random_state=42):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Initial dataset shape: {df.shape}")

    # Convert 'horsepower' to numeric, coerce errors
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    print(f"After numeric conversion, horsepower NA count: {df['horsepower'].isna().sum()}")

    # Drop rows with missing target or feature values
    df = df.dropna()
    print(f"After dropping NA rows: {df.shape}")

    # Separate X and y
    y = df['horsepower']
    X = df.drop(columns=['horsepower'])

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, drop_first=True)
    print(f"Encoded features shape: {X_encoded.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state
    )
    print(f"Train/Test split -> X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values.astype(np.float32), y_test.values.astype(np.float32)


def train_and_evaluate():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'Dados', 'Cars.csv')

    X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)
    print("Training MLPRegressor...")
    model = MLPRegressor(
        hidden_layer_sizes=(10, 5), activation='relu', solver='adam',
        max_iter=500, random_state=42
    )
    model.fit(X_train, y_train)
    print("Training complete. Evaluating...")

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print("Sklearn MLPRegressor Results:")
    print(f"  MSE: {mse:.3f}")
    print(f"  R2:  {r2:.3f}")


if __name__ == '__main__':
    train_and_evaluate()