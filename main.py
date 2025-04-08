import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# 1. 生成3D make_moons 数据
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    X += np.random.normal(scale=noise, size=X.shape)
    return X, y


# 2. 生成训练集和测试集
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

# 3. 可视化训练数据（可选）
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
ax.set_title("3D Make Moons - Training Data")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# 4. 训练和评估模型

# Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# ✅ AdaBoost + Decision Tree (新版写法)
adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100)
adaboost.fit(X_train, y_train)
y_pred_boost = adaboost.predict(X_test)

# SVM - Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

# SVM - Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)

# SVM - RBF Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

# 5. 输出评估结果
print("\n=== Decision Tree ===")
print(classification_report(y_test, y_pred_tree))

print("\n=== AdaBoost + Decision Tree ===")
print(classification_report(y_test, y_pred_boost))

print("\n=== SVM (Linear Kernel) ===")
print(classification_report(y_test, y_pred_linear))

print("\n=== SVM (Polynomial Kernel) ===")
print(classification_report(y_test, y_pred_poly))

print("\n=== SVM (RBF Kernel) ===")
print(classification_report(y_test, y_pred_rbf))
