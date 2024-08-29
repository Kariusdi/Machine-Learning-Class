import numpy as np
import matplotlib.pyplot as plt

# Function to calculate sigmoid (used for logistic regression)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def prediction(x, w0, w1):
    return w0 + w1 * x

def cost_function(x, y, w0, w1, lambda_reg):
    z = prediction(x, w0, w1)
    y_pred = sigmoid(z)
    data_loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # Regularization penalty (L2 norm of weights)
    reg_penalty = lambda_reg * (w0**2 + w1**2) / 2
    
    # Total cost with regularization
    cost = data_loss + reg_penalty
    return cost

def gradient_descent(x, y, w0, w1, learning_rate, iterations, lambda_reg):
    costs = []
    for i in range(iterations):
        z = prediction(x, w0, w1)
        y_pred = sigmoid(z)
        
        # Gradients with regularization
        w0_gradient = -np.mean(y - y_pred) - lambda_reg * w0
        w1_gradient = -np.mean(x * (y - y_pred)) - lambda_reg * w1
        
        # Update weights
        w0 = w0 - learning_rate * w0_gradient
        w1 = w1 - learning_rate * w1_gradient
        
        cost = cost_function(x, y, w0, w1, lambda_reg)
        costs.append(cost)
    
    return w0, w1, costs

def main():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 0, 0, 1, 1])

    # Hyperparameters
    learning_rate = 0.1
    iterations = 1000

    # Train the model
    w0, w1, costs = gradient_descent(x, y, 0, 0, learning_rate, iterations, lambda_reg= 0.001)

    # Calculate predicted probabilities
    z = prediction(x, w0, w1)
    y_pred = sigmoid(z)

    x_plot = np.linspace(min(x), max(x), 100)
    y_plot = sigmoid(prediction(x_plot, w0, w1))
    decision_boundary = x_plot[np.argmin(np.abs(y_plot - 0.5))]

    # Print results
    print(f'Optimized w0: {w0:.2f}')
    print(f'Optimized w1: {w1:.2f}')
    print(f'Predicted probabilities: {y_pred}')

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x_plot, y_plot, color='red', label='Logistic Regression')
    plt.axvline(decision_boundary, color='green', linestyle='--', label='Decision Boundary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Logistic Regression with Gradient Descent')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()