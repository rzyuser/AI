import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


class Canny_Edge_Detection(object):
    """ Canny 边缘检测 """
    def __init__(self, path):
        self.path = path
        # 将图片进行灰度化
        self.img_gray = self.img_chuli()
        # 高斯平滑
        self.img_new = self.Gaosi_Pinghua()
        # 求梯度
        self.img_tidu, self.tan = self.Tidu()
        # 极大值抑制
        self.img_yizhi = self.Jidazhi_Yizhi()
        # 双阈值检测
        self.Shuangyuzhi()

    def img_chuli(self):
        """ 将图片进行灰度化 """
        img = cv2.imread(self.path)
        # print(img)
        # cv2读取的时候不需要，plt读取的时候需要
        # if self.path[-4:] == '.png':
        #     img *= 255
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def Gaosi_Pinghua(self):
        """ 对图片进行高斯平滑，以上两步均是对性能优化 """
        sigma = 0.5  # 标准差
        dim = 5  # 高斯核，一般是5*5、7*7
        # 计算5*5的高斯核
        Gaussian_filter = np.zeros((dim, dim))
        tem = [i - dim // 2 for i in range(dim)]
        n1 = 1 / (2 * math.pi * sigma ** 2)
        n2 = -1 / (2 * sigma ** 2)
        for i in range(dim):
            for j in range(dim):
                Gaussian_filter[i, j] = n1 * math.exp(n2*(tem[i] ** 2 + tem[j] ** 2))
        Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
        # 进行平滑滤波
        dh, dw = self.img_gray.shape
        img_new = np.zeros([dh, dw])
        tem = dim // 2
        # 边缘填充 第一个(tem, tem) 表示上下 第二个表示左右  constant 表示填充数字默认为0
        img_pad = np.pad(self.img_gray, ((tem, tem), (tem, tem)), 'constant')
        for i in range(dh):
            for j in range(dw):
                img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * Gaussian_filter)
        return img_new

    def Tidu(self):
        """ 求梯度 """
        # 以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        dh, dw = self.img_new.shape
        print(dh, dw)
        img_tidu_x = np.zeros(self.img_new.shape)   # X方向的梯度
        img_tidu_y = np.zeros(self.img_new.shape)    # Y方向的梯度
        img_tidu = np.zeros(self.img_new.shape)         # 总的梯度
        img_pad = np.pad(self.img_new, ((1, 1), (1, 1)), 'constant')
        for i in range(dh):
            for j in range(dw):
                img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_x)
                img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_y)
                img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
        img_tidu_x[img_tidu_x == 0] = 0.00000001
        # 求每个点的梯度
        tan = img_tidu_y / img_tidu_x
        # cv2.imshow('Sobel边缘检测', img_tidu.astype(np.uint8))
        # cv2.waitKey(0)
        return img_tidu, tan

    def Jidazhi_Yizhi(self):
        """ 极大值抑制 """
        dh, dw = self.img_tidu.shape
        img_yizhi = np.zeros(self.img_tidu.shape)
        for i in range(1, dh-1):
            for j in range(1, dw-1):
                flag = True       # 是否要抑制这个点
                temp = self.img_tidu[i-1:i+2, j-1:j+2]
                if self.tan[i, j] <= -1:
                    num1 = (temp[0, 1] - temp[0, 0]) / self.tan[i, j] + temp[0, 1]
                    num2 = (temp[2, 1] - temp[2, 2]) / self.tan[i, j] + temp[2, 1]
                    if not (self.img_tidu[i, j] > num1 and self.img_tidu[i, j] > num2):
                        flag = False
                elif self.tan[i, j] >= 1:
                    num1 = (temp[0, 2] - temp[0, 1]) / self.tan[i, j] + temp[0, 1]
                    num2 = (temp[2, 0] - temp[2, 1]) / self.tan[i, j] + temp[2, 1]
                    if not (self.img_tidu[i, j] > num1 and self.img_tidu[i, j] > num2):
                        flag = False
                elif self.tan[i, j] > 0:
                    num1 = (temp[0, 2] - temp[1, 2]) * self.tan[i, j] + temp[1, 2]
                    num2 = (temp[2, 0] - temp[1, 0]) * self.tan[i, j] + temp[1, 0]
                    if not (self.img_tidu[i, j] > num1 and self.img_tidu[i, j] > num2):
                        flag = False
                elif self.tan[i, j] < 0:
                    num1 = (temp[1, 0] - temp[0, 0]) * self.tan[i, j] + temp[1, 0]
                    num2 = (temp[1, 2] - temp[2, 2]) * self.tan[i, j] + temp[1, 2]
                    if not (self.img_tidu[i, j] > num1 and self.img_tidu[i, j] > num2):
                        flag = False
                if flag:
                    img_yizhi[i, j] = self.img_tidu[i, j]
        # cv2.imshow('极大值抑制', img_yizhi.astype(np.uint8))
        # cv2.waitKey(0)
        return img_yizhi

    def Shuangyuzhi(self):
        """ 双阈值检测 """
        lower_boundary = self.img_tidu.mean() * 0.5
        # 设置高阈值是低阈值的三倍
        high_boundary = lower_boundary * 3
        zhan = []
        dh, dw = self.img_yizhi.shape
        for i in range(1, dh - 1):
            for j in range(1, dw - 1):
                if self.img_yizhi[i, j] >= high_boundary:
                    self.img_yizhi[i, j] = 255
                    zhan.append([i, j])
                elif self.img_yizhi[i, j] <= lower_boundary:
                    self.img_yizhi[i, j] = 0
        while not len(zhan) == 0:
            temp1, temp2 = zhan.pop()   # 出栈
            temp = self.img_yizhi[temp1 - 1:temp1 + 2, temp2 - 1:temp2 + 2]
            if (temp[0, 0] < high_boundary) and (temp[0, 0] > lower_boundary):
                self.img_yizhi[temp1 - 1, temp2 - 1] = 255  # 这个像素点标记为边缘
                zhan.append([temp1 - 1, temp2 - 1])  # 进栈
            if (temp[0, 1] < high_boundary) and (temp[0, 1] > lower_boundary):
                self.img_yizhi[temp1 - 1, temp2] = 255
                zhan.append([temp1 - 1, temp2])
            if (temp[0, 2] < high_boundary) and (temp[0, 2] > lower_boundary):
                self.img_yizhi[temp1 - 1, temp2 + 1] = 255
                zhan.append([temp1 - 1, temp2 + 1])
            if (temp[1, 0] < high_boundary) and (temp[1, 0] > lower_boundary):
                self.img_yizhi[temp1, temp2 - 1] = 255
                zhan.append([temp1, temp2 - 1])
            if (temp[1, 2] < high_boundary) and (temp[1, 2] > lower_boundary):
                self.img_yizhi[temp1, temp2 + 1] = 255
                zhan.append([temp1, temp2 + 1])
            if (temp[2, 0] < high_boundary) and (temp[2, 0] > lower_boundary):
                self.img_yizhi[temp1 + 1, temp2 - 1] = 255
                zhan.append([temp1 + 1, temp2 - 1])
            if (temp[2, 1] < high_boundary) and (temp[2, 1] > lower_boundary):
                self.img_yizhi[temp1 + 1, temp2] = 255
                zhan.append([temp1 + 1, temp2])
            if (temp[2, 2] < high_boundary) and (temp[2, 2] > lower_boundary):
                self.img_yizhi[temp1 + 1, temp2 + 1] = 255
                zhan.append([temp1 + 1, temp2 + 1])

        for i in range(dh):
            for j in range(dw):
                if self.img_yizhi[i, j] != 0 and self.img_yizhi[i, j] != 255:
                    self.img_yizhi[i, j] = 0

        cv2.imshow('Canny', self.img_yizhi.astype(np.uint8))
        cv2.waitKey(0)


if __name__ == '__main__':
    path = '../imgs/lenna.png'
    cannys = Canny_Edge_Detection(path)

    # 调用接口 一行解决
    cv2.Canny('灰度图', ('低阈值', '高阈值'))