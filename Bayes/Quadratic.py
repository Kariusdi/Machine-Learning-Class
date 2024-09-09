import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_classification, make_moons
from scipy.stats import norm

# Generate sample data
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
#X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
#X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
#print(X)

means = [np.mean(X[y == k], axis=0) for k in np.unique(y)]
covariances = [np.cov(X[y == k].T) for k in np.unique(y)]
priors = [np.mean(y == k) for k in np.unique(y)]

#Mean cov
def pooled_covariance(X, y):

    classes = np.unique(y)
    n_features = X.shape[1]
    pooled_cov = np.zeros((n_features, n_features))
    n_total = 0

    for c in classes:
        X_c = X[y == c]
        n_c = X_c.shape[0]
        cov_c = np.cov(X_c.T)
        pooled_cov += (n_c - 1) * cov_c 
        n_total += n_c - 1

    return pooled_cov / n_total
cov = pooled_covariance(X, y)

x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)

# Calculate the likelihood for each feature
# Mean
#likelihoods_feature1 = [norm.pdf(x_range, loc=means[k][0], scale=np.sqrt(cov[0, 0])) for k in range(len(means))]
#likelihoods_feature2 = [norm.pdf(y_range, loc=means[k][1], scale=np.sqrt(cov[1, 1])) for k in range(len(means))]
# Pool 
#likelihoods_feature1 = [norm.pdf(x_range, loc=means[k][0], scale=np.sqrt(covariances[k][0, 0])) for k in range(len(means))]
#likelihoods_feature2 = [norm.pdf(y_range, loc=means[k][1], scale=np.sqrt(covariances[k][1, 1])) for k in range(len(means))]

def qda_discriminant(x, mean, cov, prior):
    inv_cov = np.linalg.inv(cov)
    log_det_cov = np.log(np.linalg.det(cov))
    return -0.5 * np.dot(np.dot((x - mean).T, inv_cov), (x - mean)) - 0.5 * log_det_cov + np.log(prior)

def qda_predict(X):
    discriminants = Discriminants(X)
    return np.argmax(discriminants, axis=1)

def Discriminants(X):
    discriminants = np.array([
        [qda_discriminant(x, means[k], covariances[k], priors[k]) for k in range(len(means))]
        for x in X
    ])
    return discriminants

def qda_posterior(X):
    discriminants = Discriminants(X)
    max_discriminants = np.max(discriminants, axis=1, keepdims=True)
    exp_discriminants = np.exp(discriminants - max_discriminants)
    posteriors = exp_discriminants / np.sum(exp_discriminants, axis=1, keepdims=True)
    return posteriors


# Create a grid of points
xx, yy = np.meshgrid(x_range,y_range)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = qda_predict(grid).reshape(xx.shape)

posterior_feature1 = qda_posterior(np.c_[x_range, np.zeros_like(x_range)])
posterior_feature2 = qda_posterior(np.c_[np.zeros_like(y_range), y_range])

#PLOTING LIKELIHOOD
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

#PLOT A POSTERIOR
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
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


# PLOTING DECISION
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=100, cmap='viridis')
plt.scatter([mean[0] for mean in means], [mean[1] for mean in means], c='red', marker='x')

plt.title('QDA Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
