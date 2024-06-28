import numpy as np
import cv2


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    # 计算左上，右下
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, dst):
    rect = order_points(dst)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (maxWidth, maxHeight))
    return warped


if __name__ == '__main__':
    img_src = cv2.imread('./images/receipt.jpg')
    print(img_src.shape)
    ratio = img_src.shape[0] / 500
    orig = img_src.copy()

    img_src = resize(img_src, height=500)
    img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_src_gauss = cv2.GaussianBlur(img_src_gray, (5, 5), 0)
    img_src_canny = cv2.Canny(img_src_gauss, 75, 200)

    cnts = cv2.findContours(img_src_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        # 计算轮廓近似
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            # print(screenCnt)
            break

    cv2.drawContours(img_src, [screenCnt], -1, (0, 255, 0), 2)
    # 寻找角点坐标和透视变换
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # 二值化import numpy as np
    # import cv2
    #
    #
    # def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    #     dim = None
    #     (h, w) = image.shape[:2]
    #     if width is None and height is None:
    #         return image
    #     if width is None:
    #         r = height / float(h)
    #         dim = (int(w * r), height)
    #     else:
    #         r = width / float(w)
    #         dim = (width, int(h * r))
    #     resized = cv2.resize(image, dim, interpolation=inter)
    #     return resized
    #
    #
    # def order_points(pts):
    #     rect = np.zeros((4, 2), dtype="float32")
    #     s = pts.sum(axis=1)
    #     # 计算左上，右下
    #     rect[0] = pts[np.argmin(s)]
    #     rect[2] = pts[np.argmax(s)]
    #     # 计算右上和左下
    #     diff = np.diff(pts, axis=1)
    #     rect[1] = pts[np.argmin(diff)]
    #     rect[3] = pts[np.argmax(diff)]
    #     return rect
    #
    #
    # def four_point_transform(image, dst):
    #     rect = order_points(dst)
    #     (tl, tr, br, bl) = rect
    #
    #     # 计算输入的w和h值
    #     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    #     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    #     maxWidth = max(int(widthA), int(widthB))
    #
    #     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    #     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    #     maxHeight = max(int(heightA), int(heightB))
    #
    #     dst = np.array([
    #         [0, 0],
    #         [maxWidth - 1, 0],
    #         [maxWidth - 1, maxHeight - 1],
    #         [0, maxHeight - 1]], dtype='float32')
    #     m = cv2.getPerspectiveTransform(rect, dst)
    #     warped = cv2.warpPerspective(image, m, (maxWidth, maxHeight))
    #     return warped
    #
    #
    # if __name__ == '__main__':
    #     img_src = cv2.imread('./images/receipt.jpg')
    #     print(img_src.shape)
    #     ratio = img_src.shape[0] / 500
    #     orig = img_src.copy()
    #
    #     img_src = resize(img_src, height=500)
    #     img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    #     img_src_gauss = cv2.GaussianBlur(img_src_gray, (5, 5), 0)
    #     img_src_canny = cv2.Canny(img_src_gauss, 75, 200)
    #
    #     cnts = cv2.findContours(img_src_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    #     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    #
    #     for c in cnts:
    #         # 计算轮廓近似
    #         peri = cv2.arcLength(c, True)
    #         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #         if len(approx) == 4:
    #             screenCnt = approx
    #             # print(screenCnt)
    #             break
    #
    #     cv2.drawContours(img_src, [screenCnt], -1, (0, 255, 0), 2)
    #     # 寻找角点坐标和透视变换
    #     warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    #     # 二值化
    #     warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    #     ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
    #
    #     # 展示结果
    #     cv2.imshow("Original", resize(orig, height=650))
    #     cv2.imshow("Scanned", resize(ref, height=650))
    #     cv2.waitKey(0)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]

    # 展示结果
    cv2.imshow("Original", resize(orig, height=650))
    cv2.imshow("Scanned", resize(ref, height=650))
    cv2.waitKey(0)