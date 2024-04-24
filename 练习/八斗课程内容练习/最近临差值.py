import numpy as np
import cv2


img = cv2.imread('./imgs/lenna.png')

height, width, channels = img.shape
print(height, width)
img_jinlin = np.zeros((600,600, channels), np.uint8)
hl = 600 / height
wl = 600 / width

for i in range(600):
    for j in range(600):
        x = int(i / hl + 0.5)
        y = int(j / wl + 0.5)
        img_jinlin[i, j] = img[x, y]


cv2.imshow("nearest interp",img_jinlin)
cv2.imshow("src",img)
cv2.waitKey(0)