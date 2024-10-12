import numpy as np

def gradientDescent(n_samples, lr, X, y, y_pred):
    dw = 1 / n_samples * np.dot(X.T, (y_pred - y)) 
    new_weights = lr * dw 
    return new_weights

def cost_function(n_samples, y, y_pred):
    return - np.mean(np.sum(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred)))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
class LogisticRegression():
    
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.cost_history = []
        
    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(self.n_iters):
            y_pred = self.predict(X)
            cost = cost_function(n_samples, y, y_pred)
            self.cost_history.append(cost)
            new_weights = gradientDescent(n_samples, self.lr, X, y, y_pred)
            self.weights -= new_weights
            
    def predict(self, X):
        z = np.dot(X, self.weights)
        y_pred = sigmoid(z)
        return y_pred
    
    def errorRate(self, FP, FN, n_samples):
        return round(((FP + FN) / n_samples) * 100, 2)
    
    def accuracy(self, TP, TN, n_samples):
        return round(((TP + TN) / n_samples) * 100, 2)
    
    def precision(self, TP, FN):
        return round((TP / (TP + FN)) * 100, 2)
    
    def recall(self, TP, FP):
        return round((TP / (TP + FP)) * 100, 2)
    
    def specificity(self, TN, FN):
        return round((TN / (TN + FN)) * 100, 2)
    
    def get_cost_history(self):
        return self.cost_history
    
    def get_weights(self):
        return self.weights[1:]
    
    def get_weights_image(self, weights):
        # Reshape weights to 28x28 for MNIST
        weights_image = weights.reshape(28, 28)
        return weights_image
        