import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

mean1 = np.array([-1, -1])
mean2 = np.array([1, 1])
Sigma = np.array([[1, 0], [0, 1]])

n1 = 500
n2 = 500
n = n1 + n2
prior_c1 = n1 / n
prior_c2 = n2 / n

x1 = np.random.multivariate_normal(mean=mean1, cov=Sigma, size=n1)
x2 = np.random.multivariate_normal(mean=mean2, cov=Sigma, size=n2)

feature_range_x1 = np.linspace(-6, 6, 200)
feature_range_x2 = np.linspace(-6, 6, 200)

x = np.vstack([x1, x2])
y = np.concatenate([np.repeat(1, n1), np.repeat(2, n2)])

mu_hat_1 = 1 / n1 * np.sum(x1, axis=0)
mu_hat_2 = 1 / n2 * np.sum(x2, axis=0)

cov_hat_1 = 1 / (n1 - 1) * np.matmul((x1 - mu_hat_1).T, (x1 - mu_hat_1))
cov_hat_2 = 1 / (n2 - 1) * np.matmul((x2 - mu_hat_2).T, (x2 - mu_hat_2))
cov_hat = (cov_hat_1 + cov_hat_2) / 2

def likelihood(x, mu, sigma):
    """Univariate likelihood calculation for each dimension"""
    return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

def gaussian_pdf(r, mu, Sigma):
    """Multivariate Gaussian PDF"""
    D = len(mu)
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    diff = r - mu
    exp_term = np.exp(-0.5 * np.dot(np.dot(diff.T, Sigma_inv), diff))
    normalization_const = 1 / np.sqrt((2 * np.pi)**D * Sigma_det)
    return normalization_const * exp_term

def Linear_Distriminant(x, mu, sigma, prior):
    dist = x - mu
    cov_inv = np.linalg.inv(sigma)
    cov_det = np.linalg.det(sigma)
    return -1/2 * np.log(2 * np.pi * cov_det) - 1/2 * np.matmul(np.matmul(dist.T, cov_inv), dist) + np.log(prior)

def abline(x, slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    # x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x
    plt.plot(x, y_vals, '--')
    
cov_inv = np.linalg.inv(cov_hat)

# slope
slope_vec = np.matmul(cov_inv, (mu_hat_1 - mu_hat_2))
slope = -slope_vec[0] / slope_vec[1]


# intercept
intercept_partial = np.log(prior_c2) - np.log(prior_c1) + 0.5 * np.matmul(np.matmul(mu_hat_1.T, cov_inv), mu_hat_1) - 0.5 * np.matmul(np.matmul(mu_hat_2.T, cov_inv), mu_hat_2)
intercept = intercept_partial / slope_vec[1]

sigma1 = np.sqrt(Sigma[0, 0])  # standard deviation for the first dimension
sigma2 = np.sqrt(Sigma[1, 1])

# likelihood 
likelihood_c1 = np.prod([likelihood(feature_range_x1, mu, sigma1) for mu in mean1], axis=0)
likelihood_c2 = np.prod([likelihood(feature_range_x2, mu, sigma2) for mu in mean2], axis=0)

# posterior 
pdf_1 = np.array([gaussian_pdf([x, x], mean1, Sigma) for x in feature_range_x1])
pdf_2 = np.array([gaussian_pdf([x, x], mean2, Sigma) for x in feature_range_x2])

point_grid = np.mgrid[-10:10.1:0.5, -10:10.1:0.5].reshape(2, -1).T
ll_vals_1 = [Linear_Distriminant(x, mu_hat_1, cov_hat, prior_c1) for x in point_grid]
ll_vals_2 = [Linear_Distriminant(x, mu_hat_2, cov_hat, prior_c2) for x in point_grid]

posterior_c1 = pdf_1 * prior_c1
posterior_c2 = pdf_2 * prior_c2

# posteriors 
post_sum = posterior_c1 + posterior_c2
posterior_c1 /= post_sum
posterior_c2 /= post_sum


def contour_LDA(means, cov, slope, intercept, mu_hat_1, mu_hat_2 ):
    random_seed = 1000
    x = np.linspace(-4, 4, num=100)
    y = np.linspace(-4, 4, num=100)
    X, Y = np.meshgrid(x, y)
    abline(x, slope, intercept)
    plt.plot(mu_hat_1[0], mu_hat_1[1], 'rp', markersize=14)
    plt.plot(mu_hat_2[0], mu_hat_2[1], 'rp', markersize=14)
    

    for mean in means:
        # Generating a Gaussian bivariate distribution with given mean and covariance matrix
        distr = multivariate_normal(cov=cov, mean=mean, seed=random_seed)
        pdf = distr.pdf(np.dstack((X, Y)))
        plt.contour(X, Y, pdf, cmap='viridis')

    plt.colorbar(label='Density')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Combined Contour Plot of Two Gaussian Distributions")
    plt.grid(True)
    plt.show()

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(feature_range_x1, likelihood_c1, label='Class $c_1$ (Setosa)', color='black')
plt.plot(feature_range_x2, likelihood_c2, label='Class $c_2$ (Virginica)', color='green', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('Likelihood')
plt.legend()

# Posterior Probability
plt.subplot(2, 1, 2)
plt.plot(feature_range_x1, posterior_c1, label='Class $c_1$ (Setosa)', color='black')
plt.plot(feature_range_x2, posterior_c2, label='Class $c_2$ (Virginica)', color='green', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('Posterior Probability')
plt.legend()

plt.tight_layout()
plt.show()

means = [mean1, mean2]
plot = contour_LDA(means, Sigma, slope, intercept, mu_hat_1, mu_hat_2 )



