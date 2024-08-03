import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


# 详细版灰度化 cv版本读取
cv_img = cv2.imread('../imgs/lenna.png')
# cv_img1 = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
height, width = cv_img.shape[:2]
img_gray = np.zeros([height,width], cv_img.dtype)
for i in range(height):
    for j in range(width):
        m = cv_img[i,j]
        # 将BGR坐标转化为gray坐标并赋值给新图像
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
# cv2.imshow('src2', img_gray)
#
# #接口版灰度化
# img_gray1 = rgb2gray(cv_img)
# cv2.imshow('src3', img_gray1)


# plt版本
plt_img = plt.imread('./imgs/lenna.png')
plt.subplot(221)
plt.imshow(plt_img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')


# cv2.waitKey(0)


# 二值化
img_gray = rgb2gray(cv_img)
print(img_gray)
img_gray1 = np.where(img_gray >= 0.45, 1, 0)

plt.subplot(223)
plt.imshow(img_gray1, cmap='gray')

plt.show()