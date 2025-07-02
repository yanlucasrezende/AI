import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['horsepower']).dropna()
    X = df.drop(columns=['horsepower'])
    y = df['horsepower']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_and_evaluate(csv_path):
    X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)
    model = MLPRegressor(
        hidden_layer_sizes=(10, 5),
        activation='relu', solver='adam',
        max_iter=500, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Sklearn MLPRegressor Results:")
    print(f"MSE: {mean_squared_error(y_test, preds):.3f}")
    print(f"R2:  {r2_score(y_test, preds):.3f}")


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'Dados', 'Cars.csv')
    train_and_evaluate(csv_path)
