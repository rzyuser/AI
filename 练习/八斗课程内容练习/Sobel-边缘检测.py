import cv2


img = cv2.imread('./imgs/lenna.png',0)

# 边缘检测
X = cv2.Sobel(img, cv2.CV_16S, 1, 0)
Y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

# 将其转回原来的uint8，否则图像无法显示
absx = cv2.convertScaleAbs(X)
absy = cv2.convertScaleAbs(Y)

# 将其组合起来
result = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

cv2.imshow("X", absx)
cv2.imshow("Y", absy)
cv2.imshow("XY", result)

cv2.waitKey(0)
# cv2.destroyAllWindows()