import numpy as np
import cv2


def nearest_interp(img):
    """
    最近临插值计算
    :param img: 输入的图像
    :return:
    """

    # 获取图像的行、列、通道数
    height, width, channels = img.shape
    # 定义最后要输出的图像大小
    dst_h, dst_w = 1500, 1500
    # 创建一个全为0的矩阵，用来保存最后生成图像的像素
    img_zjl = np.zeros([dst_h, dst_w, channels], np.uint8)
    # 就算原图与最后生成图像的行列比例
    sh = dst_h / height
    sw = dst_w / width
    # 计算最后生成图像每个位置的像素
    for i in range(dst_h):
        for j in range(dst_w):
            y = int(i / sh + 0.5)
            x = int(j / sw + 0.5)
            # 防止图像过度放大，边缘超出界限
            if x > 511:
                x = 511
            if y > 511:
                y = 511
            # print(y,x)
            img_zjl[i, j] = img[y, x]
    return img_zjl


img = cv2.imread('../imgs/lenna.png')
imh_zjl = nearest_interp(img)
cv2.imshow('src', img)
cv2.imshow('dst', imh_zjl)
cv2.waitKey(0)
