import numpy as np


def APCA(X, K):
    # 中心化
    # mean = np.array([np.mean(i) for i in X.T])
    # center_x = X - mean
    center_x = X - X.mean(axis=0)
    # 求协方差矩阵
    C = np.dot(center_x.T, center_x) * 1 / np.shape(X)[0]
    # 求协方差矩阵的特征值和特征向量
    a, b = np.linalg.eig(C)
    # 对特征值进行从大到小排序，并选取前K个
    ids = np.argsort(-1 * a)
    U = b[:, ids[:K]]
    # UT形成的结果是行，需要使用transpose转置
    # UT = [b[:, ids[i]] for i in range(K)]
    # U2 = np.transpose(UT)
    # 求降维矩阵，原矩阵*选取的前K个特征向量组成的矩阵。
    Z = np.dot(X, U)
    return Z


X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])
K = np.shape(X)[1] - 1
# pca = APCA(X, K)
# print(pca)
# 使用接口
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=K)
pca.fit(X)
new_X = pca.fit_transform(X)
# explained_variance_ratio_:代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。
print(pca.explained_variance_ratio_)
print(new_X)
