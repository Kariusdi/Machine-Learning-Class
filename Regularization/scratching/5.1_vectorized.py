import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.RidgeRegression import RidgeRegression

if __name__ == "__main__":
    
    # df = pd.read_csv('../dataset/HeightWeight.csv')
    # first_5_rows = df.head(100)
    # Sample data
    # X = first_5_rows["Height"].values
    # y = first_5_rows["Weight"].values
    
    # Example data that easy to visualize the diff between slope of each lambda
    X = np.array([[0], [2]])
    y = np.array([0, 2])
    
    # Use Ridge regression with lambda equals to 0.5
    model = RidgeRegression(lambda_param=0.5)
    
    # Standardize data to avoid loss value to be NaN
    X = model.standardization(X)
    
    # Add bias term to X at the first column
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Train model with Gradient Descent
    model.training(X_b, y, "gradientDes")
    
    # Make a prediction
    y_pred = model.prediction(X_b)
    
    # Calculate and print the loss of the model
    error = model.costFunction(X_b, y, y_pred)
    print("Model Loss:", error)
    
    # Plot prediction line
    plt.figure(figsize=(10, 6))
    
    # Plot original data points
    plt.scatter(X, y, color='blue', label='Data points')
    # Plot prediction line
    plt.plot(X, y_pred, color='red', label='Prediction line')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Ridge Regression')
    plt.legend()
    
    # Use 5 lambda examples to see the diff
    lambdas = [0, 10, 20, 40, 400]
    # Generate slope value from -10.1 to 10.1
    slope_vals = np.arange(-10.1, 10.1, 0.1)
    
    # Add bias term for every slopes (known as weights)
    slopes = np.array([[1, v] for v in slope_vals])
    
    # Plot RSS + lambda * slope (y axis) with slopes (x axis)
    plt.figure(figsize=(10, 6))
    
    for i, lambda_param in enumerate(lambdas):
        w_history = []
        c_history = []
        for j, slope_param in enumerate(slopes):  
            # Use weights from weight genarator for visualization
            model = RidgeRegression(lambda_param=lambda_param, weights=slope_param)
            # Get cost function from every prediction from each slope and each lambda
            y_pred = model.prediction(X_b)
            loss = model.costFunction(X_b, y, y_pred)
            w_history.append(slope_param[1])
            c_history.append(loss)
        # Plot the bowl plot to see the optimum slope
        plt.plot(w_history, c_history, label=f"λ {lambda_param}")

    plt.ylim(0, 20)
    plt.xlim(-10, 12)
    plt.xlabel('Slope Value')
    plt.ylabel('Sum of Squared Residuals + λ x Slope ^ 2')
    plt.title('Ridge Regression Visualization')
    plt.legend()
    
    plt.show()