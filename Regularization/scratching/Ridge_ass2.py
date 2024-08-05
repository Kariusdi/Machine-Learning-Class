import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Sample data (replace with your data)
data = {'X1': [1, 2, 3, 4, 5],
        'X2': [2, 4, 5, 4, 5],
        'y': [7, 10, 13, 12, 14]}
df = pd.DataFrame(data)

# Split data into features (X) and target variable (y)
X = df[['X1', 'X2']]
y = df['y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Range of alpha values to explore
alpha_values = np.logspace(-4, 2, 100)

# Initialize lists to store SSR values
train_error = []
test_error = []

# Loop through alpha values
for alpha in alpha_values:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Calculate mean squared error (MSE) on training and test sets
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_error.append(mean_squared_error(y_train, train_pred))
    test_error.append(mean_squared_error(y_test, test_pred))

# Find the index of the minimum test error
best_alpha_index = np.argmin(test_error)

# Plot the SSR curve
plt.plot(alpha_values, train_error, label='Training error')
plt.plot(alpha_values, test_error, label='Test error')
plt.axvline(x=alpha_values[best_alpha_index], color='red', linestyle='--', label='Best alpha')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Ridge Regression: MSE vs Alpha')
plt.legend()
plt.show()
