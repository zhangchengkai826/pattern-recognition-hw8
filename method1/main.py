# 方案一
# 使用卷积神经网络（CNN）对选票中的三种“图像图案”进行自动分类统计
# 由于原始图像大小不一，在预处理阶段先将图像的大小统一
# CNN中的卷积层（convolution layer）负责提取特征，池化层（pooling layer）负责选择特征，最后的全连接层（fully-connected layer）负责分类
# 将一组图片随机划分为两组：训练集和测试集
# 在训练集上训练CNN，每完成一轮训练，就在测试集上进行识别分类统计，测试模型分类的准确率，当准确率不再提升时结束训练
# 具体的实现细节见下

import tensorflow as tf
from keras import datasets, layers, models
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

INPUT_SIZE = 32
TRAIN_PERCENTAGE = 0.7
NUM_CATEGORIES = 3
VERBOSITY = 2

if __name__ == '__main__':
    data_dir = 'data'
    im_prefix = 'data_binary'
    label_file_name = 'label.txt'
    images = []
    labels = []
    for file_name in os.listdir(data_dir):
        if file_name.startswith(im_prefix):
            # 对原始图像做预处理（统一大小，转化为黑白图）
            img = np.array(Image.open("{}\\{}".format(data_dir, file_name), 'r').convert('L').resize((INPUT_SIZE, INPUT_SIZE)))
            img[img > 0] = 1
            img = np.array([img])
            images.append(img)
        elif file_name == label_file_name:
            for l in open('{}\\{}'.format(data_dir, file_name), 'r'):
                lbl = [0] * 3
                lbl[int(l)] = 1
                labels.append(lbl)
    # plt.imshow(images[0])
    # plt.show()
    # print(labels)
    # print(images[0])
    
    # 建立CNN模型：卷积层-池化层-卷积层-全连接层
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 1)))
    model.add(layers.MaxPooling2D((4, 4)))
    model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    train_len = int(len(images) * TRAIN_PERCENTAGE)
    train_images = np.array(images[:train_len]).reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)
    train_labels = np.array(labels[:train_len]).reshape(-1, NUM_CATEGORIES)
    test_images = np.array(images[train_len:]).reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)
    test_labels = np.array(labels[train_len:]).reshape(-1, NUM_CATEGORIES)	
    es = EarlyStopping(monitor='val_acc', verbose=VERBOSITY, patience=50)
    history = model.fit(train_images, train_labels, epochs=1000, verbose=VERBOSITY, validation_data=(test_images, test_labels), callbacks=[es])

    # 输出训练过程中分类准确率的变化情况
    plt.plot(history.history['acc'], label='train accuracy')
    plt.plot(history.history['val_acc'], label = 'test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.title('Training Process')
    plt.show()

    # 在测试集上测试模型分类的准确率
    # 因为数据量太小（总共只有81张图片），所以模型出现了过拟合的情况（随着训练的进行，模型在训练集上的分类准确率达到100%，但在测试集上的分类准确率一直维持在60%-80%左右）
    # 两个解决方案：
    # 1. 可以通过旋转、反转、不规则缩放、加噪声等方法扩增训练集；
    # 2. 也可以使用另外的模式识别方案（详见方案二）
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=VERBOSITY)
    print("Accuracy: {}".format(test_acc))