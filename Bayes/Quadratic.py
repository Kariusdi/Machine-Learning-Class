import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_classification, make_moons

# Generate sample data
#X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
#X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
print(X)

means = [np.mean(X[y == k], axis=0) for k in np.unique(y)]
covariances = [np.cov(X[y == k].T) for k in np.unique(y)]
priors = [np.mean(y == k) for k in np.unique(y)]

def qda_discriminant(x, mean, cov, prior):
    inv_cov = np.linalg.inv(cov)
    log_det_cov = np.log(np.linalg.det(cov))
    return -0.5 * np.dot(np.dot((x - mean).T, inv_cov), (x - mean)) - 0.5 * log_det_cov + np.log(prior)

def qda_predict(X):
    discriminants = np.array([
        [qda_discriminant(x, means[k], covariances[k], priors[k]) for k in range(len(means))]
        for x in X
    ])
    return np.argmax(discriminants, axis=1)

# Create a grid of points
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = qda_predict(grid).reshape(xx.shape)


##################################################################################
# PLOTING
##################################################################################
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=100, cmap='viridis')
plt.title('QDA Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

