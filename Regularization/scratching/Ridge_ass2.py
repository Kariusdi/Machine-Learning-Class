import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def polyRidge(degree):
    # Create polynomial features
    poly = PolynomialFeatures(degree)  # Adjust degree as needed
    X_poly = poly.fit_transform(X)

    # Split data for KFold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  # Adjust n_splits

    # Initialize variables
    mse_train_list = []
    mse_test_list = []

    # Perform KFold cross-validation
    for train_index, test_index in kf.split(X_poly):
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and train the Ridge regression model
        ridge = Ridge(alpha=100000)  # Adjust alpha for regularization strength
        ridge.fit(X_train, y_train)

        # Make predictions on the test and training sets
        y_pred_train = ridge.predict(X_train)
        y_pred_test = ridge.predict(X_test)

        # Evaluate the model
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        mse_train_list.append(mse_train)
        mse_test_list.append(mse_test)

    # Estimate bias (average training error)
    E_train = np.sqrt(np.mean(mse_train_list))

    # Estimate variance (average difference between training and test error)
    E_var = np.sqrt(np.mean(np.square(np.array(mse_test_list) - np.mean(mse_train_list))))

    # Estimated E_out (sum of bias and variance)
    E_out = E_train + E_var

    # Print results and return errors
    return E_train, E_out

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
E_out_list = []
E_in_List = []

# Call the function for each degree and store errors
for deg in degrees:
  E_out, E_in = polyRidge(deg)
  E_out_list.append(E_out)
  E_in_List.append(E_in)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(degrees, E_out_list, label="E_Out", marker='o', linestyle='-')
plt.plot(degrees, E_in_List, label="E_In", marker='s', linestyle='-')
plt.xlabel("Model Complexity (Degree)")
plt.ylabel("RMSE")
plt.title("Model Complexity vs. E_train and E_test")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()