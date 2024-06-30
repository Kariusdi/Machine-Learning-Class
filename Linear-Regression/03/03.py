import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([0, 2])
y = np.array([0, 2])

# Initial parameters
w0 = 0  # Intercept
w1 = 0  # Slope
w0_history = []
w1_history = []
mse_history = []

alpha = [0.1, 0.5, 0.8, 1.001]  # Learning rate
iterations = 30  # Number of iterations

def prediction(x, w0, w1):
    return w0 + w1 * x

def gradient_descent(x, y, w0, w1, alpha, iterations):
    length = len(y)
    w1_history.append(w1)
    mse = mean_squared_error(x, y, w0, w1)
    mse_history.append(mse)

    for _ in range(iterations):
        h = prediction(x, w0, w1)
        w1 -= alpha * (1/length) * np.sum((h - y) * x)
        w1_history.append(w1)
        mse = mean_squared_error(x, y, w0, w1)
        mse_history.append(mse)
    return w0, w1

def mean_squared_error(x, y, w0, w1):
    predictions = prediction(x, w0, w1)
    mse = np.mean((predictions - y) ** 2) / 2
    return mse


fig, graph = plt.subplots(1, 4, figsize=(20, 6)) # size = 1 row 4 column 20*6

for i, alpha in enumerate(alpha): # enumerate = ดึงทั้งค่า index and value
    w1_history = []
    mse_history = []
    mse0 = []
    w1_linspace = np.linspace(-2, 4, 100) # สร้างเส้นเนื่องจาก w1_history มีการเปลี่ยน learnningRate  
    for w1_value in w1_linspace:
        mse0.append(mean_squared_error(x, y, 0, w1_value)) #error จาก w1_linspace

    # Perform gradient descent starting with w1 = -4 for each alpha
    w0, w1 = gradient_descent(x, y, 0, -2, alpha, iterations)
    graph[i].plot(w1_linspace, mse0, label='Theoretical MSE')
    graph[i].plot(w1_history, mse_history, '-o', color="red", label='Gradient Descent')
    graph[i].set_title(f'Alpha = {alpha}')
    graph[i].set_xlabel('w1')
    graph[i].set_ylabel('MSE')
    graph[i].grid(True)
    graph[i].legend()

plt.suptitle('Different Learning Rates')
plt.show()