import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Initialize the data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Feature matrix
y = np.array([0, 0, 0, 1])  # Labels (target)

# Step 2: Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Step 3: Make predictions
predictions = model.predict(X)
print("Predictions:", predictions)

# Step 4: Display final parameters (theta)
theta_final = np.hstack([model.intercept_, model.coef_.flatten()])
print("Final parameters (theta):", theta_final)

# Step 5: Evaluate the model
accuracy = accuracy_score(y, predictions)
print("Accuracy:", accuracy)

# Step 6: Plotting the decision boundary
def plot_decision_boundary(X, y, model):
    # Plot data points
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', marker='o', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='x', label='Class 1')

    # Plot decision boundary
    x_values = [np.min(X[:, 0] - 0.1), np.max(X[:, 1] + 0.1)]
    y_values = -(theta_final[0] + np.dot(theta_final[1], x_values)) / theta_final[2]
    plt.plot(x_values, y_values, label='Decision Boundary', color='green', linestyle='--')

    # Define plot attributes
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Logistic Regression with Decision Boundary')
    plt.show()

# Call the function to plot the decision boundary
plot_decision_boundary(X, y, model)



