import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('./imgs/lenna.png')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# dst = cv2.equalizeHist(img_gray)
#
# # 直方图
# hist = cv2.calcHist([dst], [0], None, [256], [0,256])
# plt.figure()
# plt.hist(dst.ravel(), 256)
# plt.show()
#
# cv2.imshow("直方图", np.hstack([img_gray, dst]))
# cv2.waitKey(0)


"""
彩色直方图均衡化
"""
img = cv2.imread('./imgs/lenna.png')
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

bgr = cv2.merge((bH, gH, rH))
cv2.imshow("dst", bgr)
cv2.waitKey(0)

