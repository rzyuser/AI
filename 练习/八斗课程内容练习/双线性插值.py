import cv2
import numpy as np


def bilinear_interp(img):
    """
    :param img: 原图
    :return: 目标图
    """
    src_h, src_w, channle = img.shape
    dst_h, dst_w = 800, 800
    bilinear_interp1 = np.zeros([dst_h, dst_w, channle], np.uint8)
    for i in range(channle):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 将图片中心化
                src_y = (dst_y + 0.5) * float(src_h) / dst_h - 0.5
                src_x = (dst_x + 0.5) * float(src_w) / dst_w - 0.5

                # 计算其周围四个点坐标
                x0 = int(src_x)
                x1 = min(x0 + 1, src_w - 1)
                y0 = int(src_y)
                y1 = min(y0 + 1, src_h - 1)

                # 双线性插值计算
                Q1 = (x1 - src_x) * img[y0, x0, i] + (src_x - x0) * img[y0, x1, i]
                Q2 = (x1 - src_x) * img[y1, x0, i] + (src_x - x0) * img[y1, x1, i]
                P = (y1 - src_y) * Q1 + (src_y - y0) * Q2
                bilinear_interp1[dst_y, dst_x, i] = int(P)
    return bilinear_interp1


img = cv2.imread('./imgs/lenna.png')
bilinear_interp1 = bilinear_interp(img)
cv2.imshow("scr", img)
cv2.imshow("dst", bilinear_interp1)
cv2.waitKey(0)