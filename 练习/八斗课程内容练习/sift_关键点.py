import cv2
import matplotlib.pyplot as plt


img = cv2.imread('./imgs/lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建sitf方法
sift = cv2.xfeatures2d.SIFT_create()
# 检测特征和计算方向 keypoints:关键点      descriptor：描述符
keypoints, descriptor = sift.detectAndCompute(img_gray, None)
# 画图
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))


cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()