# Pon
import numpy as np
import matplotlib.pyplot as plt

def standardization(X):
    mean_x = np.array([np.mean(X)])
    std_x = np.array([np.std(X)])
    X_sd = (X - mean_x) / std_x
    return X_sd

def F1_score(y, y_pred):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i] == 0 and y_pred[i] == 0:
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score

def init_theta(X):
    theta = np.array(np.zeros(X))
    return theta

def logistic_model(X, theta):
    z = np.dot(X, theta)
    y_pred = sigmoid(z)
    return y_pred

def sigmoid(z):
    sig = 1 / (1 + np.e**(-z))
    return sig

def cost_function(y, y_pred):
    cost = np.dot(-y.T, np.log(y_pred)) - np.dot((1-y).T, np.log(1 - y_pred))
    return cost 

def update_weight(n, old_weight, X, y, y_pred, lr=0.2):
    error = y_pred - y
    new_weight = old_weight - ( (lr/n) * (np.dot(X.T, error)) )
    return new_weight

def gradient_descent(n, X, y, theta, steps):
    cost_history = []
    Y_pred = logistic_model(X, theta)
    theta_list = []
    cost_history.append(1e10)

    for _ in range(1, steps+1):
        Y_pred = logistic_model(X, theta)

        cost = cost_function(y, Y_pred)
        cost_history.append(cost)
        
        theta = update_weight(n, theta, X, y, Y_pred)
        theta_list.append(theta)

    cost_history.pop(0)            
        
    return cost_history, theta_list

if __name__ == "__main__":
    steps = 100

    # AND Gate
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 0, 0, 1])

    # OR Gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    # XOR Gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    X_interaciton = np.c_[X, X[:, 0] * X[:, 1]]

    X_b = np.c_[np.ones((len(X_interaciton), 1)), X_interaciton]

    n = len(X)

    theta = init_theta(X_b.shape[1])

    cost_history, theta_list = gradient_descent(n, X_b, y, theta, steps)
    theta = theta_list[-1]

    Y_pred = logistic_model(X_b, theta)

    print(Y_pred)

    for i in range(len(Y_pred)):
        if Y_pred[i] > 0.5:
            Y_pred[i] = 1
        else:
            Y_pred[i] = 0

    print(Y_pred)

    z_values = np.linspace(-6, 6, 100)

    # # Compute the sigmoid of z
    sigmoid_values = sigmoid(z_values)

    # # Plot the sigmoid function
    plt.plot(z_values, sigmoid_values, color='blue')

    # # Add labels and a grid
    plt.axvline(0, color='black')  # Add a vertical line at z = 0
    plt.axhline(0.5, color='black')  # Add a horizontal line at y = 0.5
    plt.xlabel('z')
    plt.ylabel('Sigmoid(z)')

    plt.grid(True)

    # # Show the plot
    plt.show()