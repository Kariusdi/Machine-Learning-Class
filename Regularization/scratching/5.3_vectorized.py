import numpy as np
import matplotlib.pyplot as plt
from model.RidgeRegression import RidgeRegression
from model.LinearRegression import LinearRegression

# Define X data (bias + weigths)
def define_X_include_bias(X):
    X_b = np.c_[np.ones((len(X), 1)), X]
    return X_b

# Define the problem function
def problem(X):
    return np.sin(np.dot(np.pi, X))

# Define a mean model calculation
def meanModel(models):
    return np.mean(models, axis=0)

# Define a bias calculation
def computeBias(mean_model):
    z = np.square(mean_model - y)
    return np.mean(z)

# Define a variance calculation
def computeVariance(E_d, mean_model):
    z = np.square(E_d - mean_model)
    var_x = np.mean(z)
    return np.mean(var_x)

# Define a E out calculation
def computeEout(bias, variance):
    return bias + variance

if __name__ == "__main__":
    
    # Generate X from -1 to 1 with a hundred values
    X = np.linspace(-1, 1, 100)
    
    # Make y values from problem function (sin)
    y = problem(X)

    # Add bias term to X at the first column
    X_b = define_X_include_bias(X)
    
    # For collecting prediction value of each genaral data for both models
    E_d_linear = []
    E_d_ridge = []
    
    # Initialize subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Add the base plot to both subplots
    ax1.plot(X, y, label='sin(pi x)')
    ax1.set_xlim(-1, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('sin(pi x)')
    ax1.set_ylim(-2, 2)

    ax2.plot(X, y, label='sin(pi x)')
    ax2.set_xlim(-1, 1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('sin(pi x)')
    ax2.set_ylim(-2, 2)

    # Perform the training and plotting in the loop
    for _ in range(1000):
        # Create Linear model
        linearModel = LinearRegression()
        # Create Ridge model
        ridgeModel = RidgeRegression(lambda_param=0.5)
        
        # Randomly select 2 values from the dataset
        rands_X = np.random.choice(X, 2)
        
        # Create X and y from genaral value for training
        X_sample = define_X_include_bias(rands_X)
        y_sample = problem(rands_X)
        
        # Train model to fit the data with normal equation
        linearModel.training(X=X_sample, y=y_sample, type="normalEq")
        ridgeModel.training(X=X_sample, y=y_sample, type="normalEq")
        
        # Make a prediction with the model that train only 2 data but use with all data for both models
        y_pred_lin = linearModel.prediction(X_b)
        E_d_linear.append(y_pred_lin)
        
        y_pred_ridge = ridgeModel.prediction(X_b)
        E_d_ridge.append(y_pred_ridge)
        
        # Plot the prediction line for each iters with random genaral data
        ax1.plot(X, y_pred_lin, c="black", alpha=0.05)
        ax2.plot(X, y_pred_ridge, c="black", alpha=0.05)

    # Calculate mean model, bias, varaince, and E out for both models to see the difference between non-regularization and regularization
    mean_linearModel = meanModel(E_d_linear)
    mean_ridgeModel = meanModel(E_d_ridge)
    
    bias_linearModel = computeBias(mean_linearModel)
    variance_linearModel = computeVariance(E_d_linear, mean_linearModel)
    eOut_linearModel = computeEout(bias_linearModel, variance_linearModel)
    
    bias_ridgeModel = computeBias(mean_ridgeModel)
    variance_ridgeModel = computeVariance(E_d_ridge, mean_ridgeModel)
    eOut_ridgeModel = computeEout(bias_ridgeModel, variance_ridgeModel)
    
    print("\nLinear regression")
    print("Bias: ", bias_linearModel)
    print("Variance: ", variance_linearModel)
    print("E out: ", eOut_linearModel)
    
    print("\nRidge regression")
    print("Bias: ", bias_ridgeModel)
    print("Variance: ", variance_ridgeModel)
    print("E out: ", eOut_ridgeModel)
    
    ax1.set_title(f'Linear Regression \n Bias = {bias_linearModel:.2f} | Varinace = {variance_linearModel:.2f} | E out = {eOut_linearModel:.2f}')
    ax2.set_title(f'Ridge Regression \n Bias = {bias_ridgeModel:.2f} | Varinace = {variance_ridgeModel:.2f} | E out = {eOut_ridgeModel:.2f}')
    
    # Plot the mean predictions line
    ax1.plot(X, mean_linearModel, c="red", label="Mean Model", linewidth=2)
    ax2.plot(X, mean_ridgeModel, c="red", label="Mean Model", linewidth=2)
    
    plt.tight_layout()
    plt.show()