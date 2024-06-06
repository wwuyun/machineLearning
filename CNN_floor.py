import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# 加载 FashionMNIST 数据集
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# 数据预处理：将像素值缩放到 [0, 1] 之间，并增加一个通道维度
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

def build_and_train_model(model, train_images, train_labels, test_images, test_labels, model_name):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.1)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'{model_name}测试集准确率:', test_acc)
    return test_acc

# 存储模型准确率的字典
model_accuracies = {}

# 浅层模型
shallow_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
shallow_acc = build_and_train_model(shallow_model, train_images, train_labels, test_images, test_labels, '浅层模型')
model_accuracies['浅层模型'] = shallow_acc

# 中等深度模型
medium_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
medium_acc = build_and_train_model(medium_model, train_images, train_labels, test_images, test_labels, '中等深度模型')
model_accuracies['中等深度模型'] = medium_acc

# 深层模型
deep_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
deep_acc = build_and_train_model(deep_model, train_images, train_labels, test_images, test_labels, '深层模型')
model_accuracies['深层模型'] = deep_acc

# 输出所有模型的准确率
for model_name, accuracy in model_accuracies.items():
    print(f'{model_name}的测试集准确率: {accuracy}')
