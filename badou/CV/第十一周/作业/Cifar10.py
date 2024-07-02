import torch
import os
import tensorflow as tf

num_classes = 10


# 设定用于训练和测试的样本量
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000

class CIFAR10Record(object):
    pass


# 读取cifar-10的数据
def read_cifar10(file_queue):
    result = CIFAR10Record()
    label_bytes = 1
    result.height=32
    result.width=32
    result.depth=3
    # 样本总数量
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    # 文件读取类，读取文件内容
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 该数组第一个元素是标签，使用strided_slice()提取数据，再使用cast()改变为int32数值类型
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # 剩下这些数据在数据集中的存储形式是d * h * w, 需要将其转化为[d,h,w]
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])
    # 将c,h,w 改为h,w,c
    result.uin8image = tf.transpose(depth_major, [1,2,0])

    return result





