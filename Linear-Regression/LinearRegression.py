import numpy as np

def gradientDescent(n_samples, lr, X, y, y_pred):
    d_weigths = (1 / n_samples) * np.dot(X.T, (y_pred - y)) # d_weights = 1 / n * ∑(y_prediction - y_actual) *  X
    d_bias = (1 / n_samples) * np.sum(y_pred - y) # d_bias = 1 / n * ∑(y_prediction - y_actual)
    
    weights_gradient = lr * d_weigths
    bias_gradient = lr * d_bias
    
    return weights_gradient, bias_gradient
    

class LinearRegression:

    def __init__(self, lr = 0.001, n_iters = 0):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = 0 # wi
        self.bias = 0 # w0
     
    # Evaluation + Optimization   
    def training(self, X, y):
        n_samples, n_features = X.shape # in the shape of n x m, which means n is number of samples and m means number of features
        # wights will be zero for all feature at the first time and the size of array depends on the total of features
        self.weights = np.zeros(n_features) 
        
        for _ in range(self.n_iters):
            y_pred = self.prediction(X)
            weights_gradient, bias_gradient = gradientDescent(n_samples, self.lr, X, y, y_pred)
            self.weights -= weights_gradient
            self.bias -= bias_gradient

    # Representation
    def prediction(self, X):
        # y = wx + b, in the form of matrix calculation y = (w * X) + b
        return np.dot(X, self.weights) + self.bias