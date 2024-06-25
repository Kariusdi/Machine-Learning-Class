import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from model.LinearRegression import LinearRegression

X_data = [[0], [200]] # X represents to be a row (samples)
y_data = [0, 2]       # Y represents to be a column (output)

X = np.array(X_data)
y = np.array(y_data)

linearNormal = LinearRegression(lr=0.1, n_iters=100) # start by learning rate sets to be 0.1 with number of 100 iterations (less data => high lr)
linearNormal.training(X, y, "normalEq")              # optimize by using normal equation
predictionsNormal = linearNormal.prediction(X)             # get the prediction output after optimization

linearGradient = LinearRegression(lr=0.1, n_iters=100)  # start by learning rate sets to be 0.1 with number of 100 iterations (less data => high lr)
X_std = linearGradient.standardization(X)               # need to standardize the data before training
linearGradient.training(X_std, y, "gradientDes")        # optimize by using normal equation
predictionsGradient = linearGradient.prediction(X_std)  # get the prediction output after optimization

# finally, plot them for the visualization
plt.figure(figsize=(8,6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.plot(X, predictionsNormal, color='black', linewidth=2, label='Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Normal Equation')
plt.subplot(1, 2, 2)
plt.scatter(X_std[:, 0], y, color = "b", marker = "o", s = 30)
plt.plot(X_std, predictionsGradient, color='black', linewidth=2, label='Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gradient Descent')
plt.show()