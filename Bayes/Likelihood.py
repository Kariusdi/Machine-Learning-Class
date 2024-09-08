import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.datasets import make_moons
# Generate sample data
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

means = [np.mean(X[y == k], axis=0) for k in np.unique(y)]
covariances = [np.cov(X[y == k].T) for k in np.unique(y)]
priors = [np.mean(y == k) for k in np.unique(y)]

# Define the range for the features
x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)

# Calculate the likelihood for each feature
likelihoods_feature1 = [norm.pdf(x_range, loc=means[k][0], scale=np.sqrt(covariances[k][0, 0])) for k in range(len(means))]
likelihoods_feature2 = [norm.pdf(y_range, loc=means[k][1], scale=np.sqrt(covariances[k][1, 1])) for k in range(len(means))]

# Plot the likelihoods for feature 1
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for k in range(len(means)):
    plt.plot(x_range, likelihoods_feature1[k], label=f'Class {k}')
plt.title('Likelihood for Feature 1')
plt.xlabel('Feature 1')
plt.ylabel('Probability Density')
plt.legend()

# Plot the likelihoods for feature 2
plt.subplot(1, 2, 2)
for k in range(len(means)):
    plt.plot(y_range, likelihoods_feature2[k], label=f'Class {k}')
plt.title('Likelihood for Feature 2')
plt.xlabel('Feature 2')
plt.ylabel('Probability Density')
plt.legend()

plt.tight_layout()
plt.show()
