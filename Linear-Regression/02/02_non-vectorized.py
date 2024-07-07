import numpy as np
import matplotlib.pyplot as plt

# ข้อมูล
x = np.array([0, 20])
y = np.array([0, 20])


# x = np.array([1,2,3,4,5])             
# y = np.array([2,3,5,7,11])

w0 = 0 # จุดตัดแกน y
w1 = 0 # ความชัน
w0_history = []
w1_history = []
mse_history = []

alpha = 0.1  # Learning rate
iterations = 100  # รอบการ train
x = (x - x.mean()) / x.std()

def prediction(x, w0, w1):
    return w0 + w1 * x

def gradient_descent(x, y, w0, w1, alpha, iterations):
    length = len(y)   
    for i in range(iterations):
        h = prediction(x, w0, w1)
        w0 -= alpha * (1/length) * np.sum(h - y)
        w1 -= alpha * (1/length) * np.sum((h - y) * x)

        w0_history.append(w0)
        w1_history.append(w1)
        mse = mean_squared_error(x, y, w0, w1)
        mse_history.append(mse)
    return w0, w1, 

def mean_squared_error(x, y, w0, w1):
    predictions = prediction(x, w0, w1)
    squareErrorSum = 0
    for i in range(len(y)):
        error = predictions[i] - y[i]
        squareErrorSum += error ** 2
    mse = (1 / (2 * len(y))) * squareErrorSum
    return mse

w0, w1 = gradient_descent(x, y, w0, w1, alpha, iterations)
print(f'Optimize w0: {w0:.2f}')
print(f'Optimize w1: {w1:.2f}')

mse = mean_squared_error(x, y, w0, w1)
print(f'Mean Squared Error: {mse:.2f}')
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data')
x_line = np.linspace(min(x), max(x), 100)
y_line = prediction(x_line, w0, w1)
plt.plot(x_line, y_line, color='blue', label=f'Prediction Line: h(x) = {w0:.2f} + {w1:.2f}x')

for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], prediction(x[i], w0, w1)], 'g--')
    
plt.title('Linear Regression with Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Create a meshgrid for w0 and w1 values
w0_values = np.linspace(-10, 20, 300)
w1_values = np.linspace(-10, 20, 300)
W0, W1 = np.meshgrid(w0_values, w1_values)
Z = np.zeros(W0.shape)
# Calculate MSE for each combination of w0 and w1
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        Z[i, j] = mean_squared_error(x, y, W0[i, j], W1[i, j])

# Plot the contour map
plt.figure(figsize=(10, 6))

contour = plt.contour(W0, W1, Z, levels=np.logspace(-2, 3, 20), cmap='viridis')
plt.scatter(w0_history, w1_history, c='red', label='Gradient Descent Path')
plt.plot(w0_history, w1_history, 'r.-')
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
plt.title('Contour plot of MSE')
plt.colorbar(contour)
plt.legend()
plt.grid(True)
plt.show()