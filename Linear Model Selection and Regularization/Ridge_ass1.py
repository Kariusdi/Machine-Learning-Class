import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 2


def ridge_regression(X, y, alpha):
    n = len(X)
    mean_X = sum(X) / n
    mean_y = sum(y) / n

    # slope (b1)
    numerator = sum((X[i] - mean_X) * (y[i] - mean_y) for i in range(n))
    denominator = sum((X[i] - mean_X) ** 2 for i in range(n)) + alpha
    b1 = numerator / denominator

    # bias (b0)
    b0 = mean_y - b1 * mean_X

    return b0, b1


def predict(X, b0, b1):
    return [b0 + b1 * x for x in X]

# SSR + λ * slope^2 
def ssr_plus_reg(X, y, slope, alpha):
    n = len(X)
    predicted_y = [slope * x + 1 for x in X]
    residuals = [y[i] - predicted_y[i] for i in range(n)]
    squared_residuals = sum(r ** 2 for r in residuals)
    alpha_slop2 = alpha * slope ** 2
    return squared_residuals + alpha_slop2


plt.figure(figsize=(14, 10))
plt.subplot(2, 3, 1)
plt.scatter(X, y, color='blue', edgecolor='k')
plt.title('Data')

alphas = [0, 10, 20, 40, 400]

# Ridge Regression for different alpha values 
for i, alpha in enumerate(alphas):
    b0, b1 = ridge_regression(X, y, alpha)
    y_pred = predict(X, b0, b1)

    plt.subplot(2, 3, i + 2)
    plt.scatter(X, y, color='blue', edgecolor='k')
    plt.plot(X, y_pred, color='red')
    plt.title(f'Ridge Regression\nλ = {alpha}')

plt.tight_layout()
plt.show()


slope_values = np.linspace(-2, 2, 100)

plt.figure(figsize=(12, 6))
for alpha in alphas:
    cost_ridge = [(slope - 0.5) ** 2 + alpha * slope ** 2 for slope in slope_values]
    plt.plot(slope_values, cost_ridge, label=f'λ = {alpha}')

plt.title('Sum of Squared Residuals + λ * Slope^2')
plt.xlabel('Slope Values')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.show()

