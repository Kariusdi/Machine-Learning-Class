import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class LDA:
    def __init__(self):
        self.means = None
        self.priors = None
        self.covariance_matrix = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        n_classes = len(class_labels)
        self.means = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        self.covariance_matrix = np.zeros((n_features, n_features))

        for idx, label in enumerate(class_labels):
            X_c = X[y == label]
            self.means[idx, :] = np.mean(X_c, axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]
            centered_data = X_c - self.means[idx, :]
            self.covariance_matrix += centered_data.T @ centered_data
        self.covariance_matrix /= (X.shape[0] - n_classes)

    def predict(self, X):
        discriminants = np.zeros((X.shape[0], len(self.means)))
        inv_cov = np.linalg.inv(self.covariance_matrix)

        for idx, mean in enumerate(self.means):
            discriminants[:, idx] = (X @ inv_cov @ mean.T) - 0.5 * (mean @ inv_cov @ mean.T) + np.log(self.priors[idx])
        return np.argmax(discriminants, axis=1)


def plot_decision_boundary(X, y, model):
    plt.figure(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('LDA Decision Boundary')
    plt.show()


iris = load_iris()
X = iris.data[:, :2]  
y = iris.target

X = X[y != 0]
y = y[y != 0]
y -= 1  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LDA()
lda.fit(X_train, y_train)

# Calculate mean and std for each class
mean1, std1 = np.mean(X_train[y_train == 0]), np.std(X_train[y_train == 0])
mean2, std2 = np.mean(X_train[y_train == 1]), np.std(X_train[y_train == 1])

x = np.linspace(min(X[:, 0]) - 8, max(X[:, 0]) + 4, 200)

# Likelihood for each class
likelihood_c1 = norm.pdf(x, mean1, std1)
likelihood_c2 = norm.pdf(x, mean2, std2) 

# Prior probabilities
prior_c1 = np.mean(y_train == 0)
prior_c2 = np.mean(y_train == 1)

# Posterior probability calculation
def gaussian_prob(x, mean, std):
    coeff = 1.0 / (np.sqrt(2 * np.pi) * std)
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return coeff * exponent

feature_range = np.linspace(np.min(X_train[:, 0]) - 8, np.max(X_train[:, 0]) + 4, 2000)
pdf_1 = gaussian_prob(feature_range, mean1, std1)
pdf_2 = gaussian_prob(feature_range, mean2, std2)

# Posterior probabilities
posterior_c1 = pdf_1 * prior_c1
posterior_c2 = pdf_2 * prior_c2

post_sum = posterior_c1 + posterior_c2
posterior_c1 /= post_sum
posterior_c2 /= post_sum

fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# Plot Likelihood
axs[0].plot(x, likelihood_c1, label='Class $c_1$ (Versicolor)', color='black')
axs[0].plot(x, likelihood_c2, label='Class $c_2$ (Virginica)', color='black', linestyle='dashed')
axs[0].set_xlabel('Feature (Sepal Length)')
axs[0].set_ylabel('Likelihood')
axs[0].legend()
axs[0].set_title('Likelihood')

# Plot Posterior Probability
axs[1].plot(feature_range, posterior_c1, label='Class $c_1$ (Versicolor)', color='black')
axs[1].plot(feature_range, posterior_c2, label='Class $c_2$ (Virginica)', color='black', linestyle='dashed')
axs[1].set_xlabel('Feature (Sepal Length)')
axs[1].set_ylabel('Posterior Probability')
axs[1].legend()
axs[1].set_title('Posterior Probability ')

plt.tight_layout()
plt.show()

# Plot Decision Boundary 
plot_decision_boundary(X_train, y_train, lda)
