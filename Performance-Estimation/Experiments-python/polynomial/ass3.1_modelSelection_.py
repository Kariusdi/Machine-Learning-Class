import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import os


def load_data(file_path):
    data = pd.read_csv(file_path)
    x = data['x'].values.reshape(-1, 1)
    if 'y' in data.columns:
        y = data['y'].values
    else:
        y = data['noisy_y'].values
    return x, y

def polynomial_regression(x, y, degree):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_pred = model.predict(x_poly)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return rmse, model

def cross_validation(x, y, degree):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    cv_scores = cross_val_score(model, x_poly, y, cv=10, scoring='neg_mean_squared_error')
    rmse_cv = np.sqrt(-cv_scores.mean())
    return rmse_cv

data_dir = './datasets'
files = os.listdir(data_dir)

degreeArray = [1, 2, 3, 4, 5, 6, 7, 8]

# Initialize results dictionary
results = {
    'noiseless': {degree: [] for degree in degreeArray},
    'noisy': {degree: [] for degree in degreeArray}
} 

# Perform experiments for each file
for file in files:
    file_path = os.path.join(data_dir, file)
    # print(f"Processing file: {file_path}")
    x, y = load_data(file_path)
    
    sample_size = len(x)
    noise_level = 'noisy' if 'noisy' in file else 'noiseless'
    
    for degree in degreeArray:
        rmse_training, _ = polynomial_regression(x, y, degree)
        rmse_cv = cross_validation(x, y, degree)
        
        results[noise_level][degree].append((sample_size, rmse_training, rmse_cv))


# print(results)
for noise_level in results:
    print(f"{noise_level} data:")
    for degree in results[noise_level]:
        print(f"Degree = {degree}")
        for result in results[noise_level][degree]:
            sample_size, rmse_training, rmse_cv = result
            print(f"Sample Size: {sample_size}, Training RMSE: {rmse_training:.4f}, CV RMSE: {rmse_cv:.4f}")
        print()
