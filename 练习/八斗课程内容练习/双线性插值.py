import cv2
import numpy as np
img = cv2.imread('./imgs/lenna.png')

src_h, src_w, channels = img.shape
dst_h, dst_w = 700, 700
if src_h == dst_h and src_w == dst_w:
    dst_img = img.copy()
dst_img = np.zeros((dst_h, dst_w, channels), dtype=np.uint8)
scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
for i in range(channels):
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            # 中心化
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5
            # 防止越界
            src_x0 = int(np.floor(src_x))
            src_x1 = min(src_x0 + 1, src_w - 1)
            src_y0 = int(np.floor(src_y))
            src_y1 = min(src_y0 + 1, src_h - 1)

            # 双线性计算
            temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
            temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
            dst_img[dst_y, dst_x, i] = (src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1

cv2.imshow('bilinear interp', dst_img)
cv2.waitKey(0)