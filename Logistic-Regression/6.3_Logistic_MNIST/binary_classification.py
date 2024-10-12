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
    # 2. Select just 2 classes
    X_1_2 = X[np.any([y == 1,y == 2], axis = 0)]
    X_1_2 = np.c_[np.ones((X_1_2.shape[0], 1)), X_1_2]
    
    y_1_2 = y[np.any([y == 1,y == 2], axis = 0)]
    print("Digit 1:", np.count_nonzero(y_1_2 == 1), "samples")
    print("Digit 2:", np.count_nonzero(y_1_2 == 2), "samples\n")
    
    # 3. Split data to train and test (70/30) for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_1_2, y_1_2, test_size=0.3, random_state=42)
    print("Total train set samples:", X_train.shape[0], "samples")
    print("Total test set samples:", X_test.shape[0], "samples\n")
    
    # 4. normalize data from 0-255 to 0-1 range
    X_train_normalized = normalize(X_train)
    X_test_normalized = normalize(X_test)
    
    # 5. rescale labels to be 0 and 1
    y_train_shifted = y_train - 1
    y_test_shifted  = y_test - 1
    
    # 6. assign final dataset
    Xtrain = X_train_normalized
    Xtest = X_test_normalized
    Ytrain = y_train_shifted
    Ytest = y_test_shifted
    # ------------------------------------------------------------------
    
    # ------------------------ Machine Learning model ------------------
    # 7. define the logistic regression
    model = LogisticRegression()
    
    # 8. optimize model to find optimal weights with train set
    model.train(Xtrain, Ytrain)
    
    # 9. see predictions with test set
    Y_pred = model.predict(Xtest)
    # ------------------------------------------------------------------
    
    # ------------------------ Visualization ---------------------------
    # 10. see result of predictions
    print(f"Original prediction result: {Y_pred}")
    
    # make threshold to define which one is 1 or 0 class
    Y_pred[Y_pred >= 0.5] = 1
    Y_pred[Y_pred < 0.5] = 0
    print(f"Prediction result: {Y_pred}")
    print(f"Class 0 (digit 1): {np.count_nonzero(Y_pred == 0)}")
    print(f"Class 1 (digit 2): {np.count_nonzero(Y_pred == 1)}")
    
    # see confusion matrix
    conf_matrix = confusion_matrix(Ytest, Y_pred)
    
    # performance estimation
    TN, FP, FN, TP = conf_matrix.ravel()
    print("\nError Rate:", model.errorRate(FP, FN, Xtest.shape[0]), "%")
    print("Accuracy:", model.accuracy(TP, TN, Xtest.shape[0]), "%")
    print("Precision:", model.precision(TP, FN), "%")
    print("Recall:", model.recall(TP, FP), "%")
    print("Specificity:", model.specificity(TN, FN), "%\n")
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='PiYG', xticklabels=['0 (number 1)', '1 (number 2)'], yticklabels=['0 (number 1)', '1 (number 2)'])
    plt.xlabel('Predicted Preference', fontsize=12)
    plt.ylabel('Actual Preference', fontsize=12)
    
    # see preduction result with actual digit img
    plt.figure(figsize=(14,8))
    for i in range(6):
        X_no_bias = X_test[:, 1:]     
        image = X_no_bias[i*2].reshape(28, 28)
        plt.subplot(2,3,i+1)
        plt.imshow(image)
        title = f"True label is: {Ytest[i*2]}, predicted as: {Y_pred[i*2]}"
        plt.title(title)
    
    # see diff weights on diff digit 
    weights = model.get_weights()
    weights_1 = model.get_weights_image(np.minimum(weights, 0))
    weights_2 = model.get_weights_image(np.maximum(weights, 0))
    
    plt.figure(figsize=(12, 6))
    # Digit 1
    plt.subplot(1, 2, 1)
    plt.imshow(weights_1, cmap='coolwarm')
    plt.colorbar()
    plt.title("Digit 1")
    # Digit 2
    plt.subplot(1, 2, 2)
    plt.imshow(weights_2, cmap='coolwarm')
    plt.colorbar()
    plt.title("Digit 2")

    plt.suptitle("Coefficient of Digit 1 and 2")
    
    ch = model.get_cost_history()
    plt.figure(figsize=(12, 6))
    plt.plot(ch)
    plt.title("Cost Function")
    plt.xlabel("Number of iterations")
    plt.ylabel("$J(w,b)$", fontsize = 17)
    
    plt.show()
    # ------------------------------------------------------------------