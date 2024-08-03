import numpy as np

def gradientDescent(n_samples, lr, X, y, y_pred):
    d_weigths = np.mean(np.dot(X.T, (y_pred - y))) # d_weights = 1 / n * ∑(y_prediction - y_actual) *  X
    d_bias = np.mean((1 / n_samples) * np.sum(y_pred - y))           # d_bias = 1 / n * ∑(y_prediction - y_actual)
    
    weights_gradient = lr * d_weigths
    bias_gradient = lr * d_bias
    
    return weights_gradient, bias_gradient

def normalEquation(X, y):
    new_weights = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y)) # w = (XT * X)^-1 * (XT * y)
    return new_weights

def costFunction(n_samples, y_pred, y):                     # Mean Sqaure Error (MSE)
    return np.mean((y_pred - y) ** 2)

class LinearRegression:

    def __init__(self, lr = 0.001, n_iters = 0):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = 0 # wi
        self.bias = 0 # w0
        self.n_samples = 0
        self.weights_history = []
        self.bias_history = []
        self.costs_history = []
    
    def standardization(self, X):       # X_std = X - mean of X / standard deviation of X
        mean_x = np.array([np.mean(X)])
        std_x = np.array([np.std(X)])
        X_std = (X - mean_x) / std_x
        return X_std
     
    # Evaluation + Optimization   
    def training(self, X, y, type):
        if (type == "gradientDes"):
            n_samples, n_features = X.shape     # in the shape of n x m, which means n is number of samples and m means number of features
            self.n_samples = n_samples          # set default number of samples
            self.weights = np.zeros(n_features) # wights will be zero for all feature at the first time and the size of array depends on the total of features
            
            for _ in range(self.n_iters):
                y_pred = self.prediction(X)
                weights_gradient, bias_gradient = gradientDescent(n_samples, self.lr, X, y, y_pred)
                self.weights -= weights_gradient
                self.bias -= bias_gradient
                self.weights_history.append(self.weights[0])
                self.bias_history.append(self.bias)
                self.costs_history.append(costFunction(self.n_samples, y_pred, y));
        elif (type == "normalEq"):
            self.weights = normalEquation(X, y)
        else:
            raise Exception("Sorry, we don't have that type of optimization.")   
    
    def training_forLR(self, X, y, lr):  
        n_samples, n_features = X.shape     # in the shape of n x m, which means n is number of samples and m means number of features
        self.n_samples = n_samples          # set default number of samples
        self.weights = np.zeros(n_features) # wights will be zero for all feature at the first time and the size of array depends on the total of features
        
        for _ in range(self.n_iters):
            y_pred = self.prediction(X)
            weights_gradient, bias_gradient = gradientDescent(n_samples, lr, X, y, y_pred)
            self.weights -= weights_gradient
            self.bias -= bias_gradient
            self.weights_history.append(self.weights[0])
            self.bias_history.append(self.bias)
            self.costs_history.append(costFunction(self.n_samples, y_pred, y));
        
    def generate_costs_forContour(self, X, y, w_range, b_range):
        W, B = np.meshgrid(w_range, b_range)
        costs_history = np.zeros(W.shape)
        
        for i in range(len(w_range)):
            for j in range(len(b_range)):
                weights = w_range[i]
                bias = b_range[j]
                y_pred = weights * X[:, 0] + bias
                costs_history[j, i] = costFunction(self.n_samples, y_pred, y)
                
        return W, B, costs_history
    
    def get_Weights_Bias_History(self):
        return self.weights_history, self.bias_history
    
    def get_Costs_History(self):
        return self.costs_history

    # Representation
    def prediction(self, X):
        # y = wx + b, in the form of matrix calculation y = (w * X) + b
        return np.dot(X, self.weights) + self.bias

class CustomLinearRegression:
    def __init__(self, n_iters = 0):
        self.n_iters = n_iters
        self.weights = 0
        self.bias = 0
        self.weights_history = []
        self.bias_history = []
        self.costs_history = []
    
    def training(self, X, y, lr):
        n_samples, n_features = X.shape     # in the shape of n x m, which means n is number of samples and m means number of features
        self.weights = np.zeros(n_features) # wights will be zero for all feature at the first time and the size of array depends on the total of features
        
        for _ in range(self.n_iters):
            y_pred = self.prediction(X, -2, 0)
            weights_gradient, bias_gradient = gradientDescent(n_samples, lr, X, y, y_pred)
            self.weights -= weights_gradient
            self.bias -= bias_gradient
            self.weights_history.append(self.weights[0])
            self.bias_history.append(self.bias)
            self.costs_history.append(costFunction(n_samples, y_pred, y));
    
    def prediction(self, X, weights, bias):
        _, n_features = X.shape             # in the shape of n x m, which means n is number of samples and m means number of features
        weights = np.zeros(n_features) # wights will be zero for all feature at the first time and the size of array depends on the total of features
        return np.dot(X, weights) + bias
    
    def costFunction(self, X, y_pred, y): 
        n_samples, n_features = X.shape             # in the shape of n x m, which means n is number of samples and m means number of features
        return (1 / (2 * n_samples)) * np.sum((y_pred - y)**2)
    
    def get_Value(self):
        return self.weights_history, self.bias_history, self.costs_history