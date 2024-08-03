import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def polyRidge(degree):
    # Create polynomial features
    poly = PolynomialFeatures(degree)  # Adjust degree as needed
    X_poly = poly.fit_transform(X)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    # Create and train the Ridge regression model
    ridge = Ridge(alpha=100)  # Adjust alpha for regularization strength
    ridge.fit(X_train, y_train)

    # Make predictions on the test and training sets
    y_pred_train = ridge.predict(X_train)
    y_pred_test = ridge.predict(X_test)

    # Evaluate the model
    # Calculate E-train and E-test (RMSE)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    # Print results and return errors
    print(f"Degree={degree}: E_train: {rmse_train:.4f}, E_test: {rmse_test:.4f}")
    return rmse_train, rmse_test

def generate_sin():
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = np.sin(X) + np.random.randn(100) / 10
    return X, y

def import_csv(path):
    df = pd.read_csv(path)
    X = df['Height'].values.reshape(-1, 1)  # Reshape for sklearn
    y = df['Weight'].values
    return X, y

X, y = import_csv("Regularization/dataset/HeightWeight.csv")

# Define a range of degrees to explore
degrees = np.arange(1, 11)  # Adjust the range as needed

# Initialize lists to store errors
E_train_list = []
E_test_list = []

# Call the function for each degree and store errors
for deg in degrees:
  rmse_train, rmse_test = polyRidge(deg)
  E_train_list.append(rmse_train)
  E_test_list.append(rmse_test)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(degrees, E_train_list, label="E_train", marker='o', linestyle='-')
plt.plot(degrees, E_test_list, label="E_test", marker='s', linestyle='-')
plt.xlabel("Model Complexity (Degree)")
plt.ylabel("RMSE")
plt.title("Model Complexity vs. E_train and E_test")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()