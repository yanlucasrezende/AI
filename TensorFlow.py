import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models
import os


def load_and_preprocess(csv_path, test_ratio=0.2, val_ratio=0.2, random_state=42):
    # Load dataset
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['horsepower']).dropna()
    X = df.drop(columns=['horsepower']).values.astype(np.float32)
    y = df['horsepower'].values.astype(np.float32)

    # Shuffle
    rng = np.random.default_rng(seed=random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split train/test
    test_size = int(len(X) * test_ratio)
    val_size = int((len(X) - test_size) * val_ratio)

    X_test = X[:test_size]
    y_test = y[:test_size]
    X_val = X[test_size:test_size + val_size]
    y_val = y[test_size:test_size + val_size]
    X_train = X[test_size + val_size:]
    y_train = y[test_size + val_size:]

    # Feature normalization using Keras layer
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(X_train)
    X_train = normalizer(X_train)
    X_val = normalizer(X_val)
    X_test = normalizer(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_and_train(X_train, y_train, X_val, y_val, input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(36, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=2
    )
    return model


def train_and_evaluate(csv_path):
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess(csv_path)
    model = build_and_train(
        X_train, y_train, X_val, y_val, input_dim=X_train.shape[1]
    )
    print("Evaluating TensorFlow model on test set...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {loss:.3f}")
    print(f"Test MAE: {mae:.3f}")


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'Dados', 'Cars.csv')
    train_and_evaluate(csv_path)