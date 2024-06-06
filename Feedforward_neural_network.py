# 前馈神经网络
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

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

# 定义神经网络模型
def create_nn_model(input_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),  # 将输入数据展平
        layers.Dense(128, activation='relu'),     # 全连接层，128个神经元，ReLU激活函数
        layers.Dropout(0.2),                      # Dropout 层，防止过拟合
        layers.Dense(10, activation='softmax')   # 输出层，10个神经元，softmax激活函数
    ])
    return model

# 创建模型实例
input_shape = (28, 28)
model = create_nn_model(input_shape)

# 编译模型
model.compile(optimizer='adam',                        # 优化器：Adam
              loss='sparse_categorical_crossentropy',  # 损失函数：交叉熵损失
              metrics=['accuracy'])                    # 评估指标：准确率

# 查看模型摘要
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 评估模型性能
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
