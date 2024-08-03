# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_imahe(path):
    # 读取图片，rgb格式
    img = plt.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy+short_edge, xx: xx+short_edge]
    return  crop_img


def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            print(i)
            images.append(i)
        images = np.array(images)
        return images


def print_answer(argmax):
    with open("../../../../data/AlexNet-Keras-master/index_word.txt", 'r', encoding='utf-8') as f:
        synset = [i.split(";")[1][:-1] for i in f.readlines()]
        print(synset)
    return synset[argmax]


