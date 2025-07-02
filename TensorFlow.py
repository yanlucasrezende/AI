import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models
import os


def load_and_preprocess(csv_path, test_ratio=0.2, val_ratio=0.2, random_state=42):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Initial dataset shape (including horsepower): {df.shape}")

    # Clean and convert 'horsepower' to numeric (coerce invalid to NaN)
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    print(f"After converting horsepower to numeric, NA count: {df['horsepower'].isna().sum()}")

    # Drop rows with missing target
    df = df.dropna(subset=['horsepower'])
    print(f"After dropping NA in 'horsepower': {df.shape}")

    # Separate features and target
    y = df['horsepower'].astype(np.float32)
    X = df.drop(columns=['horsepower'])

    # Automatically encode categorical variables and keep numeric
    X_encoded = pd.get_dummies(X, drop_first=True)
    print(f"Encoded features shape: {X_encoded.shape}")

    # Convert to numpy
    X_values = X_encoded.values.astype(np.float32)
    y_values = y.values

    # Shuffle data
    rng = np.random.default_rng(seed=random_state)
    indices = np.arange(len(X_values))
    rng.shuffle(indices)
    X_values, y_values = X_values[indices], y_values[indices]

    # Split into train/val/test
    n = len(X_values)
    test_size = int(n * test_ratio)
    val_size = int((n - test_size) * val_ratio)
    print(f"Splitting data -> test: {test_size}, val: {val_size}, train: {n - test_size - val_size}")

    X_test = X_values[:test_size]
    y_test = y_values[:test_size]
    X_val = X_values[test_size:test_size + val_size]
    y_val = y_values[test_size:test_size + val_size]
    X_train = X_values[test_size + val_size:]
    y_train = y_values[test_size + val_size:]

    # Normalize features
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(X_train)
    X_train_norm = normalizer(X_train)
    X_val_norm = normalizer(X_val)
    X_test_norm = normalizer(X_test)

    print(f"Shapes -> X_train: {X_train_norm.shape}, X_val: {X_val_norm.shape}, X_test: {X_test_norm.shape}")
    return X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test


def build_and_train(X_train, y_train, X_val, y_val, input_dim):
    print("Building and training TensorFlow model...")
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=2
    )
    print("Training complete.")
    return model


def train_and_evaluate(csv_path):
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess(csv_path)
    model = build_and_train(
        X_train, y_train, X_val, y_val, input_dim=X_train.shape[1]
    )
    print("Evaluating TensorFlow model on test set...")
    loss, mae = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test MSE: {loss:.3f}")
    print(f"Test MAE: {mae:.3f}")


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'Dados', 'Cars.csv')
    train_and_evaluate(csv_path)
