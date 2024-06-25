import numpy as np
from numpy import shape
import cv2
import random


def Gaosi_Zaosheng(img, mean, sigma, percetage):
    """
    高斯噪声
    :param img: 输入图像
    :param mean:
    :param sigma:
    :param per: 给？%的像素加噪声
    :return:
    """
    Gaosi_img = img
    Gaosi_sum = int(percetage * img.shape[0] * img.shape[1])
    for i in range(Gaosi_sum):
        Y = random.randint(0, img.shape[0] - 1)
        X = random.randint(0, img.shape[1] - 1)
        Gaosi_img[Y, X] = img[Y, X] + random.gauss(mean, sigma)
        if Gaosi_img[Y, X] < 0:
            Gaosi_img[Y, X] = 0
        elif Gaosi_img[Y, X] > 255:
            Gaosi_img[Y, X] = 255

    return Gaosi_img


def Jiaoyan_Zaosheng(img, percetage):
    Jiaoyan_img = img
    Jiaoyan_sum = int(percetage * img.shape[0] * img.shape[1])
    for i in range(Jiaoyan_sum):
        Y = random.randint(0, img.shape[0] - 1)
        X = random.randint(0, img.shape[1] - 1)
        if random.random() < 0.5:
            Jiaoyan_img[Y, X] = 0
        elif random.random() >= 0.5:
            Jiaoyan_img[Y, X] = 255

    return Jiaoyan_img


# 读取图片并转化为灰度图
img = cv2.imread('./imgs/lenna.png', 0)
gaosi_img = Gaosi_Zaosheng(img, 2, 4, 0.8)
img1 = cv2.imread('./imgs/lenna.png', 0)
jiaoyan_img = Jiaoyan_Zaosheng(img1, 0.2)
img2 = cv2.imread('./imgs/lenna.png', 0)
cv2.imshow("src", img2)
cv2.imshow("gaosi", gaosi_img)
cv2.imshow("jiaoyan", jiaoyan_img)
cv2.waitKey(0)

from skimage import util

'''
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
功能：为浮点型图片添加各种随机噪声
参数：
image：输入图片（将会被转换成浮点型），ndarray型
mode： 可选择，str型，表示要添加的噪声类型
	gaussian：高斯噪声
	localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
	poisson：泊松噪声
	salt：盐噪声，随机将像素值变成1
	pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
	s&p：椒盐噪声
	speckle：均匀噪声（均值mean方差variance），out=image+n*image
seed： 可选的，int型，如果选择的话，在生成噪声前会先设置随机种子以避免伪随机
clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。如果谁False，则输出矩阵的值可能会超出[-1,1]
mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
--------
返回值：ndarry型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数
'''

img = cv2.imread("lenna.png")
noise_gs_img = util.random_noise(img, mode='s&p', amount=0.1)

cv2.imshow("source", img)
cv2.imshow("lenna", noise_gs_img)
# cv.imwrite('lenna_noise.png',noise_gs_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
