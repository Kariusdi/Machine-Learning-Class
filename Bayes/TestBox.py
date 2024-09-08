import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scipy.stats import norm

# Generate sample data
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

means = [np.mean(X[y == k], axis=0) for k in np.unique(y)]
covariances = [np.cov(X[y == k].T) for k in np.unique(y)]
priors = [np.mean(y == k) for k in np.unique(y)]

x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)

def qda_discriminant(x, mean, cov, prior):
    inv_cov = np.linalg.inv(cov)
    log_det_cov = np.log(np.linalg.det(cov))
    return -0.5 * np.dot(np.dot((x - mean).T, inv_cov), (x - mean)) - 0.5 * log_det_cov + np.log(prior)

def qda_posterior(X):
    discriminants = np.array([
        [qda_discriminant(x, means[k], covariances[k], priors[k]) for k in range(len(means))]
        for x in X
    ])
    max_discriminants = np.max(discriminants, axis=1, keepdims=True)
    exp_discriminants = np.exp(discriminants - max_discriminants)
    posteriors = exp_discriminants / np.sum(exp_discriminants, axis=1, keepdims=True)
    return posteriors

# Calculate posterior probabilities for each feature
posterior_feature1 = qda_posterior(np.c_[x_range, np.zeros_like(x_range)])
posterior_feature2 = qda_posterior(np.c_[np.zeros_like(y_range), y_range])

# Plot posterior probabilities for each feature
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Feature 1
for i in range(2):
    ax[0].plot(x_range, posterior_feature1[:, i], label=f'Class {i}')
ax[0].set_title('Posterior Probability for Feature 1')
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Posterior Probability')
ax[0].legend()

# Feature 2
for i in range(2):
    ax[1].plot(y_range, posterior_feature2[:, i], label=f'Class {i}')
ax[1].set_title('Posterior Probability for Feature 2')
ax[1].set_xlabel('Feature 2')
ax[1].set_ylabel('Posterior Probability')
ax[1].legend()

plt.tight_layout()
plt.show()
