import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from model.LinearRegression import LinearRegression

if __name__ == "__main__":
    
    X_data = [[0], [2]] # X represents to be a row (samples)
    y_data = [0, 2]     # Y represents to be a column (output) 

    X = np.array(X_data)
    y = np.array(y_data)

    # finally, plot them for the visualization
    fig, graph = plt.subplots(1, 4, figsize=(20, 6))
    lr = [0.1, 0.5, 0.8, 1.001]
    
    def costFunction(n_samples, y_pred, y):                     # Mean Sqaure Error (MSE)
        return (1 / (2 * n_samples)) * np.sum((y_pred - y)**2)
    
    def gradientDescent(n_samples, lr, X, y, weigth, bias):
        y_pred = np.dot(X, weigth) + bias
        d_weigths = (1 / n_samples) * np.dot(X.T, (y_pred - y)) # d_weights = 1 / n * ∑(y_prediction - y_actual) *  X
        d_bias = (1 / n_samples) * np.sum(y_pred - y)           # d_bias = 1 / n * ∑(y_prediction - y_actual)
        
        weights_gradient = lr * d_weigths
        bias_gradient = lr * d_bias
        
        return weights_gradient, bias_gradient
    
    def prediction(X, weights, bias):
        # y = wx + b, in the form of matrix calculation y = (w * X) + b
        return np.dot(X, weights) + bias
    
    n_samples, n_features = X.shape 
    bias = 0
    
    linear = LinearRegression(lr=0.1, n_iters=100) # start by learning rate sets to be 0.1 with number of 100 iterations (less data => high lr)
    linear.training(X, y, "gradientDes")           # optimize by using gradient descent
    
    w_history, b_history = linear.get_Weigths_Bias_History() # get the value history for visualizing with contour
    c_history = linear.get_Costs_History()
    
    for i, lr in enumerate(lr):
        weights_range = np.linspace(-4, 5, 100)
        w1_history = []
        w0_history = []
        errors = []
        for weigth in weights_range:
            predictions = np.dot(X, weigth) + bias          # get the prediction output after optimization
            errors.append(costFunction(n_samples, predictions, y))
        
        w1, w0 = gradientDescent(n_samples, lr, X, y, 0)
        w1_history.append(w1)
        w0_history.append(w0);
        graph[i].plot(weights_range, errors)
        graph[i].plot(w_history, c_history, '-o', color="red")
            
            
    
    
    
    plt.suptitle('Different Learning Rates')
    plt.show()