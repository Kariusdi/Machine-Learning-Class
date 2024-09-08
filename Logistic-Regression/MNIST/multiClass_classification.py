import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model.LogisticRegression import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import copy

def plot_weights(weights, num_digits=10):
    plt.figure(figsize=(12, 8))
    for i in range(num_digits):
        # Extract the weights corresponding to the ith digit
        digit_weights = weights[i].reshape(28, 28)
        
        plt.subplot(2, 5, i + 1)  # Create a grid for 10 digits
        plt.imshow(digit_weights, cmap='coolwarm')
        plt.colorbar()
        plt.title(f"Digit {i}")
        plt.axis("off")
    
    plt.suptitle("Weight Visualization for All Digits")
    plt.show()

def normalize(X):
    return X / 255.0

if __name__ == "__main__":
    
    # 1. Fetch MNIST data
    mnist = fetch_openml('mnist_784', version=1)
    
    # Create 10 copies of the dataset
    datasets = [copy.deepcopy(mnist) for _ in range(10)]
    
    # print(mnist["data"].to_numpy())
    # print(datasets[0]["target"].to_numpy())
    
    # 3. Modify the target for each dataset to be binary (1 for a specific digit, 0 for others)
    for i in range(10):
        target_array = datasets[i]["target"].to_numpy().astype(int)
        binary_target = np.where(target_array == i, 1, 0)
        datasets[i]["target"] = binary_target
    
    models = []
    weights = []
    
    for i in range(10):
        X = datasets[i]["data"].to_numpy()[:5000]
        X = np.c_[np.ones((X.shape[0], 1)), X]
        Y = datasets[i]["target"][:5000]
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        # 4. normalize data from 0-255 to 0-1 range
        X_train = normalize(X_train)
        X_test = normalize(X_test)
        
        model = LogisticRegression()
        model.train(X_train, Y_train)
        models.append(model)
        weight = model.get_weights()
        weight = model.get_weights_image(weight)
        weights.append(weight)
        print(f"\nTrained model for digit {i} successfully. ✅")
    
    
    # Multiclass classification using the trained models
    X_test_all = mnist["data"].to_numpy()
    X_test_all = np.c_[np.ones((X_test_all.shape[0], 1)), X_test_all]
    X_test_all = normalize(X_test_all)
    Y_test_all = mnist["target"].to_numpy().astype(int)
    
    predictions = np.zeros((X_test_all.shape[0], 10))
    
    # Get predictions from each model
    for i in range(10):
        Y_pred = models[i].predict(X_test_all)
        predictions[:, i] = Y_pred  # Store predictions for each class
    
    # Convert predicted probabilities to class labels (take argmax across the 10 models)
    final_predictions = np.argmax(predictions, axis=1)
    
    # Evaluate the multiclass classification performance
    conf_matrix = confusion_matrix(Y_test_all, final_predictions)
    print(f"Multiclass Confusion Matrix:\n{conf_matrix}")
    
    accuracy = np.mean(final_predictions == Y_test_all)
    print(f"Multiclass Accuracy: {accuracy * 100:.2f}%")
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
        
    plot_weights(weights)
        