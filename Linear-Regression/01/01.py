import numpy as np
import matplotlib.pyplot as plt
from model.LinearRegression import LinearRegression

X_data = [[0], [2]] # X represents to be a row (samples)
y_data = [0, 2] # Y represents to be a column (output)
X = np.array(X_data)
y = np.array(y_data)

linear = LinearRegression(lr=0.1, n_iters=100) # start by learning rate sets to be 0.1 with number of 100 iterations (less data => high lr)
linear.training(X, y)                          # optimize by using gradient descent
predictions = linear.prediction(X)             # get the prediction output after optimization
w_history, b_history = linear.get_Weigths_Bias_History() # get the value history for visualizing with contour
c_history = linear.get_Costs_History()

# mock the weigths (wi) and bias (w0) up from w_history and b_history to draw the contour
weigths_range = np.linspace(-0.3, 1.5, 100)
bias_range = np.linspace(-0.3, 1.5, 100)

# and pass them to calculate the costs
W, B, cost_history = linear.generate_costs_forContour(X, y, weigths_range, bias_range)

# finally, plot them for the visualization
plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.plot(X, predictions, color='black', linewidth=2, label='Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Representation After Optimization')
plt.subplot(1, 2, 2)
plt.contour(W, B, cost_history)
plt.scatter(w_history, b_history, color = "b", marker = "o") 
plt.xlabel('weigths')
plt.ylabel('bias')
plt.title('Cost Function (MSE) Contour Plot with Gradient Descent Steps')

# visulize error reduction
plt.figure(figsize=(8,6))
plt.plot(range(100), c_history);
plt.xlabel('iterations')
plt.ylabel('error')
plt.title('Error Reduction')
plt.show()