import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# 加载 FashionMNIST 数据集
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# 数据预处理：将像素值缩放到 [0, 1] 之间，并增加一个通道维度
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

# 定义不同大小的卷积核
conv_sizes = [(3, 3), (5, 5), (7, 7)]

for size in conv_sizes:
    # 创建新的卷积神经网络模型实例
    model = models.Sequential([
        layers.Conv2D(32, size, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, size, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, size, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    print(f"Training model with kernel size: {size}")
    model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_split=0.1)

    # 在测试集上评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy with kernel size {size}: {test_acc}')