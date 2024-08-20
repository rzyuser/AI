from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers, utils
import matplotlib.pyplot as plt
import os, cv2, time

# 加载数据
(train_img, train_label), (test_img, test_label) = mnist.load_data()

# 建立模型
network = models.Sequential()
# 增加网络
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# softmax 将最终结果映射到[0,1]
network.add(layers.Dense(10, activation='softmax'))
# 编译
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 数据预处理
train_img = train_img.reshape((60000, 28*28))
train_img = train_img.astype('float32') / 255
test_img = test_img.reshape((10000, 28*28))
test_img = test_img.astype('float32') / 255
train_label = utils.to_categorical(train_label)
test_label = utils.to_categorical(test_label)

# 训练
network.fit(train_img, train_label, epochs=6, batch_size=128)

# 测试
test_loss, test_acc = network.evaluate(test_img, test_label, verbose=1)

#预测
img_list = os.listdir('../imgs/test_img')
for i in range(len(img_list)):
    print(img_list[i])
    dicts = cv2.imread(f'../imgs/test_img/{img_list[i]}', 0)
    plt.imshow(dicts, cmap=plt.cm.binary)
    plt.show()
    dicts = dicts.reshape((1, 28*28))
    res = network.predict(dicts)
    print(res)
    # print(res.shape)
    # print(res.shape[1])
    for j in range(res.shape[1]):
        if (res[0][j] == 1):
            print("the number for the picture is : ", j)
            time.sleep(4)