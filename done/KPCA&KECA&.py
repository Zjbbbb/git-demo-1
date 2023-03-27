import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_kernels
from scipy.spatial.distance import pdist, squareform
from operator import itemgetter


def print_plot_2(ax, Z, n_samples, i, j):
    ax.scatter(Z[:n_samples, i], Z[:n_samples, j],
               color='blue', marker='o', alpha=0.5)
    ax.scatter(Z[n_samples:n_samples*2, i], Z[n_samples:n_samples*2, j],
               color='red', marker='o', alpha=0.5)
    ax.scatter(Z[n_samples*2:n_samples*3, i], Z[n_samples*2:n_samples*3, j],
               color='black', marker='o', alpha=0.5)


def print_plot_3(Z, n_samples):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(
        Z[:n_samples, 0],
        Z[:n_samples, 1],
        Z[:n_samples, 2],
        color='blue', marker='o', alpha=0.5
    )
    ax.scatter(
        Z[n_samples:n_samples*2, 0],
        Z[n_samples:n_samples*2, 1],
        Z[n_samples:n_samples*2, 2],
        color='red', marker='o', alpha=0.5
    )
    ax.scatter(
        Z[n_samples*2:n_samples*3, 0],
        Z[n_samples*2:n_samples*3, 1],
        Z[n_samples*2:n_samples*3, 2],
        color='black', marker='o', alpha=0.5
    )


def rbf_kpca(X, gamma, n_components):

    # # 计算成对样本之间的欧式距离
    # sq_dists = pdist(X, 'sqeuclidean')
    # # 将成对距离转换为方阵
    # mat_sq_dists = squareform(sq_dists)
    # # 计算核矩阵
    # K = np.exp(-gamma * mat_sq_dists)
    K = pairwise_kernels(X, metric='rbf', gamma=gamma)

    # 中心化核矩阵
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    # 矩阵中每一个值先减去所在行均值,再减去所在列均值,最后加上所有值得均值
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # 计算特征值和特征向量
    eigvals, eigvecs = np.linalg.eigh(K)
    # 对特征值从小到大排序
    eigValIndice = np.argsort(eigvals)
    # 最大的n个特征值的下标
    idx = eigValIndice[-1:-(n_components+1):-1]
    # 最大的n个特征值对应的特征值和特征向量
    eigvals_de = [eigvals[i] for i in range(-1, -(n_components+1), -1)]
    eigvecs_de = eigvecs[:, idx]

    # # 收集特征值前n_components个大小的特征向量
    # X_pc = [eigvecs[:, -i] for i in range(1, n_components+1)]
    # X_kpca = -1*X_pc[0]
    # for i in range(1, n_components):
    #     X_kpca = np.column_stack((X_kpca, -1*X_pc[i]))

    # 计算贡献值
    sum1 = sum(eigvals_de)
    sum2 = sum(eigvals)
    rate = sum1/sum2
    return eigvecs_de, rate

# KECA处理结果


def rbf_keca(X, gamma, k):
    # 生成核矩阵
    sq_dist = pdist(X, metric='sqeuclidean')
    mat_sq_dist = squareform(sq_dist)
    K = np.exp(-gamma*mat_sq_dist)
    # K = pairwise_kernels(X, metric='rbf', gamma=gamma)
    # kk = pairwise_kernels(X, metric='rbf', gamma=gamma)
    # print(kk)
    # step 2
    N = X.shape[0]
    one_N = np.ones((N, N))/N
    K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

    # step 3
    Lambda, Q = np.linalg.eig(K)
    eigen_pairs = [(Lambda[i], Q[:, i]) for i in range(len(Lambda))]

    # 熵值排序
    Renyi = np.zeros_like(Lambda)
    for i in range(len(Lambda)):
        # Renyi[i] = sum(abs(Q[:, i] * (Lambda[i]**(1/2))))
        Renyi[i] = np.sum(np.abs(Q[:, i] * np.sqrt(np.abs(Lambda[i]))))

    Renyi_pairs = [(Renyi[i], Lambda[i], Q[:, i]) for i in range(len(Lambda))]
    # print()
    Renyi_pairs = sorted(Renyi_pairs, reverse=True, key=itemgetter(0))

    New_Q = np.zeros_like(Q)
    for i in range(len(Lambda)):
        New_Q[i] = Renyi_pairs[i][2]

    New_Q = New_Q.T
    # 还原对角矩阵
    f = np.dot(np.dot(np.linalg.inv(New_Q), K), New_Q)
    f = np.around(f, decimals=8)

    # TargetF = np.column_stack((f[:, i].tolist() for i in range(k)))
    # TargetQ = np.column_stack((New_Q[:, i].tolist() for i in range(k)))
    TargetF = np.stack([row for row in np.transpose(f.T)])
    TargetQ = np.stack([row for row in np.transpose(New_Q.T)])

    TargetECA = np.dot(TargetF**(1/2), TargetQ.T)
    # TargetECA = np.row_stack((TargetECA[i, :] for i in range(k)))
    TargetECA = np.row_stack([TargetECA[i, :].tolist() for i in range(k)])

    return TargetECA.T


if __name__ == '__main__':
    # 定义常数
    pi = np.pi          # pi
    n_samples = 200     # 样本数量
    noise = 0.2         # 高斯白噪声标准差
    n_components = 3    # 特征值数量
    gamma = 5e-3        # KPCA核函数系数

    # 生成独立高斯白噪声
    e1 = np.random.normal(0, noise, n_samples)
    e2 = np.random.normal(0, noise, n_samples)
    e3 = np.random.normal(0, noise, n_samples)
    e4 = np.random.normal(0, noise, n_samples)
    e5 = np.random.normal(0, noise, n_samples)
    e6 = np.random.normal(0, noise, n_samples)

    # 生成x,y样本信号
    t = np.arange(0, 2*pi, 2*pi/n_samples)
    x1 = 1 * np.sin(t) + e1
    y1 = 1 * np.cos(t) + e2
    x2 = 3 * np.sin(t) + e3
    y2 = 3 * np.cos(t) + e4
    x3 = 6 * np.sin(t) + e5
    y3 = 6 * np.cos(t) + e6
    Z1 = np.column_stack((x1, y1))
    Z2 = np.column_stack((x2, y2))
    Z3 = np.column_stack((x3, y3))
    Z = np.concatenate((Z1, Z2, Z3), axis=0)

    # X_kpca, rate = rbf_kpca(Z, gamma, n_components)
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    # print_plot_2(ax[0], Z, n_samples, 0, 1)
    # print_plot_2(ax[1], X_kpca, n_samples, 0, 2)
    # print_plot_3(X_kpca, n_samples)
    # plt.show()

    X_keca = rbf_keca(Z, gamma, n_components)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    print_plot_2(ax[0], Z, n_samples, 0, 1)
    print_plot_2(ax[1], X_keca, n_samples, 0, 2)
    print_plot_3(X_keca, n_samples)
    plt.show()
