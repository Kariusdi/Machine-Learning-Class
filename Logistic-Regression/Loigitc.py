import numpy as np
import matplotlib.pyplot as plt

# Step 1: Initialize the data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Feature matrix
y = np.array([0, 0, 0, 1])  # Labels (target)

# Adding the intercept term (bias) by adding a column of ones to X
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Step 2: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 3: Cost function (Logistic loss function)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-7  # Small constant to avoid log(0)
    cost = (1/m) * (-y.T @ np.log(h + epsilon) - (1 - y).T @ np.log(1 - h + epsilon))
    return cost

# Step 4: Gradient Descent
def gradient_descent(X, y, theta, learning_rate, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        # Calculate the hypothesis
        h = sigmoid(X @ theta)
        
        # Update theta (parameter vector)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= learning_rate * gradient

        # Save the cost for each iteration
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

# Step 5: Predict function
def predict(X, theta):
    prob = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in prob]

# Step 6: Running the logistic regression
theta_initial = np.zeros(X.shape[1])
learning_rate = 0.1
num_iters = 1000

# Perform gradient descent
theta_final, cost_history = gradient_descent(X, y, theta_initial, learning_rate, num_iters)

# Display the results
print("Final parameters (theta):", theta_final)
print("Cost after training:", cost_history[-1])

# Make predictions
predictions = predict(X, theta_final)
print("Predictions:", predictions)

# Step 7: Plotting the results
def plot_decision_boundary(X, y, theta):
    # Plot data points
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], color='red', marker='o', label='Class 0')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], color='blue', marker='x', label='Class 1')

    # Plot decision boundary
    x_values = [np.min(X[:, 1] - 0.1), np.max(X[:, 2] + 0.1)]
    y_values = -(theta[0] + np.dot(theta[1], x_values)) / theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary', color='green', linestyle='--')

    # Define plot attributes
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Logistic Regression with Decision Boundary')
    plt.show()

# Call the function to plot the decision boundary
plot_decision_boundary(X, y, theta_final)
