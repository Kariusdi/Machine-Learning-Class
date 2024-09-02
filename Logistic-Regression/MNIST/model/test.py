import numpy as np
import matplotlib.pyplot as plt

def gradientDescent(n_samples, lr, X, y, y_pred, weights):
    dw = 1 / n_samples * np.dot(X.T, (y_pred - y)) 
    new_weights = lr * dw 
    return new_weights

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cost_function(n_samples, y, y_pred):
    # Cross-entropy loss for multiclass classification
    y_one_hot = np.eye(y_pred.shape[1])[y]
    return - 1.0 / n_samples * np.sum(y_one_hot * np.log(y_pred))

class LogisticRegression():
    
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.cost_history = []
        
    def train(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.weights = np.zeros((n_features, n_classes))
        y_one_hot = np.eye(n_classes)[y]
        
        for _ in range(self.n_iters):
            y_pred = self.predict(X)
            cost = cost_function(n_samples, y, y_pred)
            self.cost_history.append(cost)
            for i in range(n_classes):
                class_y = y_one_hot[:, i]
                y_pred_class = y_pred[:, i]
                new_weights = gradientDescent(n_samples, self.lr, X, class_y, y_pred_class, self.weights[:, i])
                self.weights[:, i] -= new_weights
            
    def predict(self, X):
        z = np.dot(X, self.weights)
        y_pred = softmax(z)
        return y_pred
    
    def get_cost_history(self):
        return self.cost_history
    
    def get_weights(self):
        return self.weights
    
    def get_weights_image(self, weights):
        # Reshape weights to 28x28 for MNIST
        weights_image = weights.reshape(28, 28)
        return weights_image