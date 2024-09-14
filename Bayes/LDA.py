import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X = iris.data[:, 0] 
y = iris.target

# (Versicolor, Virginica)
# X = X[y != 0]
# y = y[y != 0]
# y -= 1  

# (setosa, Versicolor)
X = X[y != 2]
y = y[y != 2]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mean1, std1 = np.mean(X_train[y_train == 0]), np.std(X_train[y_train == 0])
mean2, std2 = np.mean(X_train[y_train == 1]), np.std(X_train[y_train == 1])

def lda_fit(X, y):

    mean_vectors = [np.mean(X[y == i], axis=0) for i in np.unique(y)]
    S_W = np.zeros((1, 1))  
    for i, mean_vec in enumerate(mean_vectors):
        sum_cov = np.cov(X[y == i], rowvar=False)  # คำนวณ covariance
        S_W += sum_cov

    overall_mean = np.mean(X, axis=0)
    S_B = np.zeros((1, 1)) 
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == i].shape[0]
        mean_vec = mean_vec.reshape(1, 1)
        overall_mean = overall_mean.reshape(1, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    # หาทิศทางที่เหมาะสมที่สุด (Eigenvector) โดยแก้สมการ S_W^-1 S_B
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    
    return eig_vecs[:, np.argmax(eig_vals)], mean_vectors


def lda_predict(X, eig_vec, mean_vectors):
    
    X = X.reshape(-1, 1) 
    projections = np.dot(X, eig_vec)
    mean_projections = [np.dot(mean_vec, eig_vec) for mean_vec in mean_vectors]
    class_predictions = [0 if abs(p - mean_projections[0]) < abs(p - mean_projections[1]) else 1 for p in projections]
    
    return np.array(class_predictions)


eig_vec, mean_vectors = lda_fit(X_train, y_train)
y_pred = lda_predict(X_test, eig_vec, mean_vectors)

# accuracy
accuracy = np.mean(y_pred == y_test)
print(f'LDA Accuracy: {accuracy * 100:.2f}%')

x = np.linspace(min(X) - 1, max(X) + 1, 200)

likelihood_c1 = norm.pdf(x, mean1, std1)
likelihood_c2 = norm.pdf(x, mean2, std2)

prior_c1 = np.mean(y_train == 0) 
prior_c2 = np.mean(y_train == 1)

def gaussian_prob(x, mean, std):
    coeff = 1.0 / (np.sqrt(2*np.pi)*std)
    exponent = np.exp(-((x-mean) ** 2) / (2*std **2))
    return coeff * exponent

feature_range = np.linspace(np.min(X_train) - 2, np.max(X_train) + 2, 2000)
pdf_1 = gaussian_prob(feature_range, mean1, std1)
pdf_2 = gaussian_prob(feature_range, mean2, std2)

# คำนวณความน่าจะเป็นเชิงหลัง (Posterior)
posterior_c1 = pdf_1 * prior_c1
posterior_c2 = pdf_2 * prior_c2

post_sum = posterior_c1 + posterior_c2
posterior_c1 /= post_sum
posterior_c2 /= post_sum

plt.figure(figsize=(8, 8))

# Likelihood
plt.subplot(2, 1, 1)
plt.plot(x, likelihood_c1, label='Class $c_1$ (Versicolor)', color='black')
plt.plot(x, likelihood_c2, label='Class $c_2$ (Virginica)', color='green', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('Likelihood')
plt.legend()

# Posterior Probability
plt.subplot(2, 1, 2)
plt.plot(feature_range, posterior_c1, label='Class $c_1$ (Versicolor)', color='black')
plt.plot(feature_range, posterior_c2, label='Class $c_2$ (Virginica)', color='green', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('Posterior Probability')
plt.legend()

plt.tight_layout()
plt.show()

# decision boundary
plt.plot(posterior_c1, posterior_c2, label='Linear distriminant (LDA)', color='blue')
plt.xlabel('class1')
plt.ylabel('class2')
plt.legend()
plt.show()
