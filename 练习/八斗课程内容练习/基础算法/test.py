import numpy as np
import pylab
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n, k, t, d, debug, return_all):
    """
        ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)
        :param data:
        :param model:
        :param n:   生成模型所需的最少样本点
        :param k: 迭代次数
        :param t:   误差
        :param d:   内群停止条件
        :param debug:
        :param return_all:
        :return:
    """
    iterations = 0  # 迭代的计数
    bestfit = None  # 最佳匹配
    besterr = np.inf  # 误差，最开始设置成默认值
    best_inlier_idxs = None  # 最开始的内群
    while iterations < k:
        maybe_idxs, test_idx = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idx]
        maybemodel = model.fit(maybe_inliers)
        # 计算最小误差平方和
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idx[test_err < t]
        also_inliers = data[also_idxs, :]
        if len(also_inliers) > d:
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel(object):
    """ sl.lstsq 返回结果解析
        求解线性最小二乘问题
        最小二乘解：x 类似[[17.231]]
        残差平方和：resids
        系数矩阵秩：rank
        系数矩阵奇异值：k
    """
    def fit(self, data):
        self.A = data[:, [0]]
        self.B = np.vstack([data[:, 1]]).T
        x, resids, rank, k = sl.lstsq(self.A, self.B)
        return x    # 返回最小平方和向量

    def get_error(self, data, model):
        self.A = data[:, [0]]
        self.B = np.vstack([data[:, 1]]).T
        B_fit = np.dot(self.A, model)
        err_per_point = np.sum((self.B - B_fit) ** 2, axis=1)
        return err_per_point       # 误差平方和


def test():
    # 生成理想数据
    n_samples = 500     # 样本数量
    n_inputs = 1    # 输入的变量
    n_outputs = 1   # 输出的变量
    A_test = 20 * sp.random.random((n_samples, n_inputs))   # 生成500个0～20的5行一列的数据
    tan_x = 40 * np.random.random(size=(n_inputs, n_outputs))   # 生成1个0～40的斜率
    B_test = np.dot(A_test, tan_x)
    # 加入噪声
    A_noisy = A_test + np.random.normal(size=A_test.shape)
    B_noisy = B_test + np.random.normal(size=B_test.shape)
    # 增加离群点
    n_outliers = 100
    all_ids = np.arange(A_noisy.shape[0])
    np.random.shuffle(all_ids)
    n_ids = all_ids[:n_outliers]
    A_noisy[n_ids] = 30 * np.random.random((n_outliers, n_inputs))
    B_noisy[n_ids] = 45 * np.random.normal(size=(n_outliers, n_inputs))
    # 将A_noisy和B_noisy左右拼接起来
    all_data = np.hstack((A_noisy, B_noisy))

    model = LinearLeastSquareModel()
    linear_fit, resids, rank, s = sl.lstsq(all_data[:, [0]], all_data[:, [1]])
    # RANSAC算法
    debug = False
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)
    sort_idxs = np.argsort(A_test[:, 0])
    A_col0_sorted = A_test[sort_idxs]
    # 画图
    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='mnist_data')
    pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC mnist_data")
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, tan_x)[:, 0], label='exact system')
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')
    pylab.legend()
    pylab.show()


if __name__ == '__main__':
    test()
