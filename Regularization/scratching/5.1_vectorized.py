import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.RidgeRegression import RidgeRegression

# Example usage:
if __name__ == "__main__":
    df = pd.read_csv('../dataset/HeightWeight.csv')
    first_5_rows = df.head(100)

    # Sample data
    # X = first_5_rows["Height"].values
    # y = first_5_rows["Weight"].values
    
    X = np.array([[0], [2]])
    y = np.array([0, 2])
    
    # Create and train the model
    model = RidgeRegression(lambda_param=0)
    
    X = model.standardization(X)
    
    # Add bias term to X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    model.training(X_b, y, "gradientDes")
    
    # Make predictions
    predictions = model.predict(X_b)
    
    # Calculate and print the loss
    error = model.costFunction(X_b, y)
    print("Loss:", error)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot original data points
    plt.scatter(X, y, color='blue', label='Data points')
    
    plt.plot(X, predictions, color='red', label='Prediction line')
    
    # Labels and title
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Ridge Regression')
    plt.legend()
    
    lambdas = [0, 10, 20 , 40, 400]
    slope_vals = np.arange(-30.1, 30.1, 0.1)
    slopes = np.array([[0, v] for v in slope_vals])
    plt.figure(figsize=(10, 6))
    
    for i, lambda_param in enumerate(lambdas):
        w_history = []
        c_history = []
        for j, slope_param in enumerate(slopes):  
            model = RidgeRegression(lambda_param=lambda_param, weights=slope_param)
            loss = model.costFunction(X_b, y)
            w_history.append(slope_param[1])
            c_history.append(loss)
        plt.plot(w_history, c_history, label=f"λ {lambda_param}")

    plt.ylim(0, 20)
    plt.xlim(-10, 12)
    plt.xlabel('Slope Value')
    plt.ylabel('Sum of Squared Residuals + λ x |Slope|')
    plt.title('Ridge Regression Visualization')
    plt.legend()
    
    plt.show()