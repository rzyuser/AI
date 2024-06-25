import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)





img1 = cv2.imread('./imgs/iphone1.png')
img2 = cv2.imread('./imgs/iphone2.png')

sift = cv2.xfeatures2d.SIFT_create()

keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
keypoints2, descriptor2 = sift.detectAndCompute(img2, None)

#开始匹配，申明匹配工具里面用到的计算距离方法 NORM_L2：欧式距离
bf = cv2.BFMatcher(cv2.NORM_L2)
# knnMatch是一种蛮力匹配，全量遍历,寻找相似度最高的前两个
matches = bf.knnMatch(descriptor1, descriptor2, k=2)

goodMatch = []
# m,n表示得到一些匹配点
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1, keypoints1, img2, keypoints2, goodMatch[:20])


cv2.waitKey(0)
cv2.destroyAllWindows()