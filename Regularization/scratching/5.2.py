import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def generate_sin():
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    Y = np.sin(X) + np.random.randn(100) / 10
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return X_train, Y_train, X_test, Y_test

def import_csv(path):
    df = pd.read_csv(path)
    X = df['Height'].values.reshape(-1, 1)  # Reshape for sklearn
    Y = df['Weight'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return X_train, Y_train, X_test, Y_test

def ridge_regression(X_train, Y_train, X_test, Y_test, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X_train, Y_train)

    train_rmse = mean_squared_error(Y_train, model.predict(X_train))
    test_rmse = mean_squared_error( Y_test, model.predict(X_test))

    return train_rmse, test_rmse

def plot_rmse_vs_alpha(alphas, train_rmse, test_rmse):

    plt.figure(figsize=(8, 6))
    plt.plot(alphas, train_rmse, label="Train", marker='o', linestyle='-')
    plt.plot(alphas, test_rmse, label="Test", marker='s', linestyle='-')
    plt.xlabel("Model Complexity (Log scale)")
    plt.xscale('log')
    plt.ylabel("RMSE")
    plt.title("Model Complexity vs. E_train and E_test")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

X_train, Y_train, X_test, Y_test = import_csv("../dataset/HeightWeight.csv")
#X_train, Y_train, X_test, Y_test = generate_sin()

alphas = np.arange(1, 100000, 100)
E_train = []
E_test = []

for alpha_ in alphas:
    train_rmse, test_rmse = ridge_regression(X_train, Y_train, X_test, Y_test, alpha_)
    E_train.append(np.sqrt(train_rmse))
    E_test.append(np.sqrt(test_rmse))

plot_rmse_vs_alpha(alphas, E_train, E_test)


