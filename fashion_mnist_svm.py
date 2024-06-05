import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

# 加载FashionMNIST数据集
mnist = fetch_openml("Fashion-MNIST", version=1)
X, y = mnist["data"], mnist["target"]

# 将标签转换为整数（0-9）
y = y.astype(int)

# 数据预处理：将 DataFrame 转换为 NumPy 数组，并归一化
X_array = X.values.astype(float) / 255.0

# 数据重塑
X_reshaped = X_array.reshape(-1, 28, 28)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.3, random_state=42)

# 特征提取：使用HOG
def extract_features(X):
    features = []
    for image in X:
        hog_features = hog(image, pixels_per_cell=(14, 14), cells_per_block=(1, 1), orientations=9, block_norm='L2', visualize=False, transform_sqrt=True)
        features.append(hog_features)
    return np.array(features)

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# 初始化SVM分类器
svm_classifier = SVC(kernel='linear', C=1.0)

# 训练模型
svm_classifier.fit(X_train_features, y_train)

# 预测测试集
y_pred = svm_classifier.predict(X_test_features)

# 性能评估
print(classification_report(y_test, y_pred))

# 评估预测结果
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
