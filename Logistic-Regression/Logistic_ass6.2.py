import numpy as np
from sklearn.datasets import make_classification, make_blobs, make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=500, noise=0.2)
# X, y = make_blobs(n_samples=500, centers=2, n_features=2, cluster_std=3)
# X, y = make_classification(n_samples=400, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# cost function
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
        
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)
        
        if i % 100 == 0:
            print(f'Iteration {i}, Cost: {cost}')
    
    return weights, cost_history

def predict(X, weights):
    return sigmoid(np.dot(X, weights)) >= 0.5


def plot_decision_boundary(X, y, weights, ax):
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = predict(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()], weights)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    ax.scatter(X[:, 1], X[:, 2], c=y, edgecolors='k', marker='o', cmap='coolwarm')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Logistic Regression Decision Boundary')


# add bias
X = np.hstack((np.ones((X.shape[0], 1)), X))
weights = np.zeros(X.shape[1])

learning_rate = 0.1
iterations = 1000
weights, cost_history = gradient_descent(X, y, weights, learning_rate, iterations)


predictions = predict(X, weights)
print("Predictions:", predictions)
print("Actual values:", y)


fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot original data
axs[0].scatter(X[:, 1], X[:, 2], c=y, cmap='coolwarm')
axs[0].set_title("Data set")
axs[0].set_xlabel("Feature 1")
axs[0].set_ylabel("Feature 2")

# Plot decision boundary
plot_decision_boundary(X, y, weights, axs[1])

plt.tight_layout()
plt.show()
