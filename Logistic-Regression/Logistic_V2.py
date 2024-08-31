import numpy as np
import matplotlib.pyplot as plt

# Step 1: Initialize the data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Feature matrix
y = np.array([1, 0, 0, 1])  # Labels (target)

def standardize(x):
    xavr = np.mean(x, axis=0)
    xstd = np.std(x, axis=0)
    return (x - xavr) / xstd

def init_theta(X):
    return np.zeros(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic(X, theta):
    z = np.dot(X, theta)
    return sigmoid(z)

def compute_cost(y_predict, y):
    epsilon = 1e-7  # Small constant to avoid log(0)
    cost = np.dot(-y.T, np.log(y_predict + epsilon)) - np.dot((1 - y).T, np.log(1 - y_predict + epsilon))
    return cost

def update_weight(n, weight, X, y, y_pred, learn):
    err = np.sqrt((y_pred - y)**2)
    new = weight - ((learn/n) * (np.dot(X.T, err)))
    return new

def gradient_descent(n, X, y, theta, iter_rate):
    cost_his = []
    theta_his = []
    Y_predict = logistic(X, theta)
    cost_his.append(1e7)

    for i in range(1, iter_rate+1):
        Y_predict = logistic(X, theta)

        cost = compute_cost(Y_predict, y)
        cost_his.append(cost) 

        theta = update_weight(n, theta, X, y, Y_predict, learn)
        theta_his.append(theta)

    cost_his.pop(0)

    return cost_his, theta_his
        

if __name__ == "__main__":

    X = standardize(X)

    #interaction term
    interaction_x = np.c_[X, X[:, 0] * X[:, 1]]
    
    #Bias terms
    X_b = np.c_[np.ones((len(interaction_x), 1)), interaction_x]

    data_len = len(X)

    theta = init_theta(X_b.shape[1])

    num_iters = 1000
    learn = 0.1

    cost_his, theta_his = gradient_descent(data_len, X_b, y, theta, num_iters)

    theta = theta_his[-1]

    y_pred = logistic(X_b, theta)

    print(y_pred)

    for i in range(len(y_pred)):
        if y_pred[i] >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    print(y_pred)