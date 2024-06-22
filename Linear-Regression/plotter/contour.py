import matplotlib.pyplot as plt

def Plot_optimizationAndContour(predictions, X, y, w_history, b_history, W, B, cost_history):
    plt.figure(figsize=(15,7))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
    plt.plot(X, predictions, color='black', linewidth=2, label='Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Representation After Optimization')
    plt.subplot(1, 2, 2)
    plt.contour(W, B, cost_history)
    plt.scatter(w_history, b_history, color = "b", marker = "o") 
    plt.xlabel('weigths')
    plt.ylabel('bias')
    plt.title('Cost Function (MSE) Contour Plot with Gradient Descent Steps')
    plt.show()