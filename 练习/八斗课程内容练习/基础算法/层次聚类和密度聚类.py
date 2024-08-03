from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# 层次聚类
# X = [[1, 3], [3, 2], [4, 4], [1, 2], [1, 3]]
# Z = linkage(X, 'ward')
#
# F = fcluster(Z, 4, 'distance')
# fig = plt.figure(figsize=(10, 12))
# dn = dendrogram(Z)
# plt.show()

# 密度聚类
from sklearn.cluster import DBSCAN
from sklearn import datasets

iris = datasets.load_iris()
# print(iris)
x = iris.data[:, :4]
# print(x)
dascan = DBSCAN(eps=0.4, min_samples=10)
dascan.fit(x)
label_pred = dascan.labels_
print(label_pred)

# x_1 = x[label_pred == -1]   # -1表示异常点
x0 = x[label_pred == 0]
x1 = x[label_pred == 1]
x2 = x[label_pred == 2]
print(x0)
plt.scatter(x0[:, 0], x0[:, 1], c='red', marker='o', label='label1')
plt.scatter(x1[:, 0], x1[:, 1], c='green', marker='*', label='label2')
plt.scatter(x2[:, 0], x2[:, 1], c='blue', marker='+', label='label3')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

