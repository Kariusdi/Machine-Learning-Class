import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os


def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
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
    return model


files = {
    'Noiseless_10': './datasets/sin_noiseless_10sample.csv',
    'Noisy_10': './datasets/sin_noisy_10sample.csv',
    'Noiseless_10': './datasets/sin_noiseless_10sample.csv',
    'Noisy_10': './datasets/sin_noisy_10sample.csv',
    # 'Noiseless_20': 'Performance-Estimation/Experiments-python/polynomial/data/sin_noiseless_20sample.csv',
    # 'Noisy_20': 'Performance-Estimation/Experiments-python/polynomial/data/sin_noisy_20sample.csv',
    # 'Noiseless_40': 'Performance-Estimation/Experiments-python/polynomial/data/sin_noiseless_40sample.csv',
    # 'Noisy_40': 'Performance-Estimation/Experiments-python/polynomial/data/sin_noisy_40sample.csv',
    'Noiseless_80': './datasets/sin_noiseless_80sample.csv',
    'Noisy_80': './datasets/sin_noisy_80sample.csv'
}

degree = 8
results = {}

# Process each file and store coefficientsฆ
for key, file_path in files.items():
    x, y = load_data(file_path)
    if x is None or y is None:
        continue
    model = polynomial_regression(x, y, degree)
    coefficients = np.append(model.intercept_, model.coef_[1:])
    results[key] = coefficients


index = [f'w{i}' for i in range(1, degree+1)]
df_results = pd.DataFrame(index=index)

# Fill in the results DataFrame
for key, coeffs in results.items():
    for i in range(len(coeffs)):
        df_results.at[f'w{i}', key] = coeffs[i]
        
print(df_results)
# print(results)

