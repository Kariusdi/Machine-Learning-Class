import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# โหลดข้อมูล Iris
iris = load_iris()
X = iris.data[:, :2]  # เลือก Sepal length และ Sepal width
y = iris.target       # คลาส (0: Setosa, 1: Versicolor, 2: Virginica)

# เลือกเฉพาะคลาส 0 และ 1 (Setosa และ Versicolor)
X = X[y != 2]
y = y[y != 2]

# แยกข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# สร้าง meshgrid สำหรับการคำนวณ decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# คำนวณค่า decision boundary
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# สร้างกราฟ
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')  # แสดง Decision Boundary

# แสดงข้อมูล training
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 0 (Setosa)', edgecolor='k')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Class 1 (Versicolor)', edgecolor='k')

plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Decision Boundary and Data Scatter Plot')
plt.legend()
plt.show()
