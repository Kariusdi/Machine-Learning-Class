import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X_data = [[0], [2]]
y_data = [0, 2]
X = np.array(X_data)
y = np.array(y_data)

# print("X =", X, X.T)
# print("y =", y)

reg = LinearRegression(lr=0.01, n_iters=1000)
reg.training(X, y)
predictions = reg.prediction(X)
print(predictions)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.plot(X, predictions, color='black', linewidth=2, label='Prediction')
plt.show()

def costFunction(y, predictions):
    return np.mean((y - predictions)**2) # Mean Sqaure Error (MSE)

MSE = costFunction(y, predictions)
print(MSE)