import numpy as np
import matplotlib.pyplot as plt
from model.LinearRegression import LinearRegression
from plotter.contour import Plot_optimizationAndContour

X_data = [[0], [2]] # X represents to be a row (samples)
y_data = [0, 2] # Y represents to be a column (output)
X = np.array(X_data)
y = np.array(y_data)

linear = LinearRegression(lr=0.1, n_iters=100) # start by learning rate sets to be 0.1 with number of 100 iterations (less data => high lr)
linear.training(X, y)                          # optimize by using gradient descent
predictions = linear.prediction(X)             # get the prediction output after optimization
w_history, b_history = linear.get_Weigths_Bias_History() # get the value history for visualizing with contour

# mock the weigths (wi) and bias (w0) up from w_history and b_history to draw the contour
weigths_range = np.linspace(-0.3, 1.5, 100)
bias_range = np.linspace(-0.3, 1.5, 100)

# and pass them to calculate the costs
W, B, cost_history = linear.generate_costs_forContour(X, y, weigths_range, bias_range)

# finally, plot them for the visualization
Plot_optimizationAndContour(predictions, X, y, w_history, b_history, W, B, cost_history)