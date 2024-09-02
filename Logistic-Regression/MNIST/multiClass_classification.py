import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from model.LogisticRegression import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_digit(some_digit):
    some_digit_image = some_digit.reshape(28,28)
    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
    plt.axis("off")
    plt.show()

def normalize(X):
    return X / 255.0

if __name__ == "__main__":
    
    # 1. Fetch MNIST data
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist["data"].to_numpy()
    y = mnist["target"].astype(np.int8).to_numpy()
    print(f"\nOriginal samples : {X.shape[0]} samples")
    print(f"Original features : {X.shape[1]} features\n")
    
    # ------------------------ Data Preparation ------------------------
    # 2. Use all classes
    X = X
    y = y.astype(np.int8)

    # 3. Split data to train and test (70/30) for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Total train set samples:", X_train.shape[0], "samples")
    print("Total test set samples:", X_test.shape[0], "samples\n")

    # 4. Normalize data from 0-255 to 0-1 range
    X_train_normalized = normalize(X_train)
    X_test_normalized = normalize(X_test)

    # 5. Assign final dataset
    Xtrain = X_train_normalized
    Xtest = X_test_normalized
    Ytrain = y_train
    Ytest = y_test
    # ------------------------------------------------------------------
    
    # ------------------------ Machine Learning model ------------------
    # 6. Define the logistic regression
    model = LogisticRegression(lr=0.1, n_iters=1000)
    
    # 7. Optimize model to find optimal weights with train set
    model.train(Xtrain, Ytrain)
    
    # 8. See predictions with test set
    Y_pred_prob = model.predict(Xtest)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    # ------------------------------------------------------------------
    
    # ------------------------ Visualization ---------------------------
    # 9. See result of predictions
    print(f"Original prediction result: {Y_pred}")
    print(f"Class distribution in predictions: {np.bincount(Y_pred)}")
    
    # See confusion matrix
    conf_matrix = confusion_matrix(Ytest, Y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='PiYG', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.title('Confusion Matrix')
    plt.show()

    # Visualize some predictions with actual digit images
    plt.figure(figsize=(14,8))
    for i in range(6):
        image = Xtest[i*2].reshape(28, 28)
        plt.subplot(2, 3, i+1)
        plt.imshow(image, cmap='gray')
        title = f"True label: {Ytest[i*2]}, Predicted: {Y_pred[i*2]}"
        plt.title(title)
    
    # See weights for each class
    weights = model.get_weights()
    plt.figure(figsize=(15, 15))
    for i in range(weights.shape[1]):
        plt.subplot(4, 5, i+1)  # Adjust subplot grid size based on number of classes
        plt.imshow(weights[:, i].reshape(28, 28), cmap='coolwarm')
        plt.title(f"Digit {i}")
    plt.suptitle("Weights for Each Digit")
    plt.show()
    
    # Plot cost function history
    ch = model.get_cost_history()
    plt.figure(figsize=(12, 6))
    plt.plot(ch)
    plt.title("Cost Function")
    plt.xlabel("Number of iterations")
    plt.ylabel("$J(w,b)$", fontsize=17)
    plt.show()
    # ------------------------------------------------------------------
