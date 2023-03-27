import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_kernels
from scipy.spatial.distance import pdist, squareform
from operator import itemgetter
import time


def print_3d(map, str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = len(map)
    x = []
    y = []
    z = []
    for i in range(n):
        for j in range(n):
            x.append(i)
            y.append(j)
            z.append(map[i][j])

    ax.scatter(x, y, z, c='r', marker='o', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(str)
    plt.show()


def Generated_data(K, k, err):
    pi = np.pi          # pi

    # 均匀分布
    e1 = np.random.uniform(-0.1, 0.1, K)
    e2 = np.random.uniform(-0.1, 0.1, K)
    e3 = np.random.uniform(-0.1, 0.1, K)
    e4 = np.random.uniform(-0.1, 0.1, K)
    e5 = np.random.uniform(-0.1, 0.1, K)
    e6 = np.random.uniform(-0.1, 0.1, K)
    e7 = np.random.uniform(-0.1, 0.1, K)

    # # 高斯分布
    # noise = 0.1
    # e1 = np.random.normal(0, noise, K)
    # e2 = np.random.normal(0, noise, K)
    # e3 = np.random.normal(0, noise, K)
    # e4 = np.random.normal(0, noise, K)
    # e5 = np.random.normal(0, noise, K)
    # e6 = np.random.normal(0, noise, K)
    # e7 = np.random.normal(0, noise, K)

    x1 = 0.5*k*k - 2*k + 0.5 + e1
    x2 = k*k - k + np.sin(k*pi) + e2

    x3 = np.zeros_like(x1)
    for i in range(K):
        if i < 3:
            x3[i] = 0.5*x2[i-3+K] - 1.01*x1[i-4+K] + 0.1*np.cos(pi*k) + e3[i]
        elif i < 4:
            x3[i] = 0.5*x2[i-3] - 1.01*x1[i-4+K] + 0.1*np.cos(pi*k) + e3[i]
        else:
            x3[i] = 0.5*x2[i-3] - 1.01*x1[i-4] + 0.1*np.cos(pi*k) + e3[i]

        # 添加阶跃变化
        if 200 <= i < 300:
            x3[i] = x3[i] + err

    x4 = np.zeros_like(x1)
    for i in range(K):
        if i < 1:
            x4[i] = -0.22*x2[i-1+K]*x1[i] - k + 5 + e4[i]
        else:
            x4[i] = -0.22*x2[i-1]*x1[i] - k + 5 + e4[i]

    x5 = np.zeros_like(x1)
    for i in range(K):
        if i < 2:
            x5[i] = (x3[i-2+K])**2 - 0.47*x1[i] + 2*k + e5[i]
        else:
            x5[i] = (x3[i-2])**2 - 0.47*x1[i] + 2*k + e5[i]

    x6 = np.zeros_like(x1)
    for i in range(K):
        if i < 1:
            x6[i] = x5[i-1+K] + 6.6 + e6[i]
        else:
            x6[i] = x5[i-1] + 6.6 + e6[i]

    x7 = np.zeros_like(x1)
    for i in range(K):
        if i < 4:
            x7[i] = x6[i-4+K] - 0.47*x4[i] - k + e7[i]
        else:
            x7[i] = x6[i-4] - 0.47*x4[i] - k + e7[i]

    X = np.column_stack((x1, x2, x3, x4, x5, x6, x7))
    return X


def Sliding_window(array, L):
    # 获取原始数组的大小
    K, n = array.shape

    # 创建一个用于存储数据的新数组
    new_arr = np.zeros((K-L+1, n, L))

    # 切分滑动窗口并将其放入输出数组中
    for i in range(K-L+1):
        new_arr[i] = array[i:i+L].T

    # 将L*n的窗口数组转换成1*(L*n)的一维数组
    new_arr = new_arr.reshape(K-L+1, n, -1)

    # 转置输出数组以匹配想要的形状
    new_arr = new_arr.transpose((0, 2, 1))

    # 创建一个用于返回的新数组，里面是1*(K-L+1)个数组
    result = [[] for _ in range(K-L+1)]

    # 将新数组里面的每个数据从多个小矩阵变成一个大一维数组
    for i in range(K-L+1):
        result[i] = np.array(new_arr[i].reshape(1, n*L))

    # 将数组转换成矩阵返回
    result = np.concatenate(result, axis=0).reshape((K-L+1, n*L))
    return result


def rbf_keca(X, gamma, k):
    # 生成核矩阵
    sq_dist = pdist(X, metric='sqeuclidean')
    mat_sq_dist = squareform(sq_dist)
    K = np.exp(-gamma*mat_sq_dist)
    # K = pairwise_kernels(X, metric='rbf', gamma=gamma)

    N = X.shape[0]
    one_N = np.ones((N, N))/N
    K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

    # step 3
    Lambda, Q = np.linalg.eig(K)
    eigen_pairs = [(Lambda[i], Q[:, i]) for i in range(len(Lambda))]

    # 熵值排序
    Renyi = np.zeros_like(Lambda)
    for i in range(len(Lambda)):
        Renyi[i] = np.sum(np.abs(Q[:, i] * np.sqrt(np.abs(Lambda[i]))))

    Renyi_pairs = [(Renyi[i], Lambda[i], Q[:, i]) for i in range(len(Lambda))]
    Renyi_pairs = sorted(Renyi_pairs, reverse=True, key=itemgetter(0))

    New_Q = np.zeros_like(Q)
    for i in range(len(Lambda)):
        New_Q[i] = Renyi_pairs[i][2]

    New_Q = New_Q.T
    # 还原对角矩阵
    f = np.dot(np.dot(np.linalg.inv(New_Q), K), New_Q)
    f = np.around(f, decimals=8)

    TargetF = np.stack([row for row in np.transpose(f.T)])
    TargetQ = np.stack([row for row in np.transpose(New_Q.T)])

    TargetECA = np.dot(TargetF**(1/2), TargetQ.T)
    TargetECA = np.row_stack([TargetECA[i, :].tolist() for i in range(k)])

    return TargetECA.T


def keca_dissim(K, k, n, L, gamma, n_components, err):

    # 生成数据集
    # 最后 X 是一个(n*K)*7的数组。
    # 其中 X[:K,] 是第一批数据集
    X = Generated_data(K, k, 0)
    if err == 0:
        for i in range(1, n):
            X = np.concatenate((X, Generated_data(K, k, 0)))
    else:
        n_half = int(n/2)
        for i in range(1, n_half):
            X = np.concatenate((X, Generated_data(K, k, 0)))
        for i in range(n_half, n):
            X = np.concatenate((X, Generated_data(K, k, err)))

    # 计算数据切片
    # 最后 X_Slide 是一个(n1*(K-L+1))*(7*L)。
    # 其中 X_Slide[:K,] 是第一批训练集
    X_Slide = Sliding_window(X[:K,], L)
    for i in range(1, n):
        X_Slide = np.concatenate(
            (X_Slide, Sliding_window(X[(K*i):(K*i+K),], L)))

    # 进行KECA降维
    # 最后 X_Slide_KECA 是一个 (n*(K-L+1))*n_components 的数组。
    # 其中 X_Slide_KECA[:(K-L+1),] 是第一批训练集对应的降维后的KECA矩阵
    X_Slide_KECA = rbf_keca(X_Slide[:(K-L+1),], gamma, n_components)
    for i in range(1, n):
        X_Slide_KECA = np.concatenate(
            (X_Slide_KECA,
             rbf_keca(X_Slide[(K-L+1)*i:((K-L+1)*(i+1)),],
                      gamma, n_components)
             ))

    return X_Slide_KECA


if __name__ == '__main__':
    # 定义常数
    pi = np.pi          # pi
    K = 400             # 每组的样本数量 400
    Gaussian_noise = 0.2  # 高斯白噪声标准差
    n_components = 3    # 特征值数量
    gamma = 5e-3        # KPCA核函数系数
    L = 4               # 数据切片长度
    k = 0.5             # 常数,取值范围[-2,2]
    n1 = 100            # 训练集个数 100
    n2 = 400            # 测试集个数 400
    n2_half = int(n2/2)  # 半个测试集个数
    Start_time = time.time()  # 程序开始时间

    # 生成训练集以及训练集KECA后的矩阵
    # 是一个 (n1*(K-L+1))*n_components 的数组
    X_train_Slide_KECA = keca_dissim(
        K, k, n1, L, gamma, n_components, 0)

    # 计算训练集各个批次之间的欧氏距离相异度 Distances_train 是 n1*n1 的矩阵
    Distances_train = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(i+1, n1):
            Distances_temp = np.linalg.norm(X_train_Slide_KECA[(K-L+1)*i:((K-L+1)*(i+1)),] -
                                            X_train_Slide_KECA[(K-L+1)*j:((K-L+1)*(j+1)),], axis=1)
            Distances_train[i][j] = sum(Distances_temp)
            Distances_train[j][i] = sum(Distances_temp)

    # print_3d(Distances_train, 'Distances_train')

    # 计算每个批次与其他批次之间的 KECA-DISSIM 指标之和
    Distances_train_sum = np.sum(Distances_train, axis=1)

    # 中心、边界批次索引
    min_index = np.argmin(Distances_train_sum)
    max_index = np.argmax(Distances_train_sum)

    # 中心、边界批次矩阵。目前边界批次矩阵只有一个
    # 最后是一个 (K-L+1)*n_components 数组
    Center_vector = X_train_Slide_KECA[((K-L+1)*min_index):
                                       ((K-L+1)*(min_index+1)), ]
    Boundary_vector = X_train_Slide_KECA[((K-L+1)*max_index):
                                         ((K-L+1)*(max_index+1)), ]

    Boundary_C_B = np.linalg.norm(Boundary_vector - Center_vector, axis=1)
    Boundary_C_B_sum = sum(Boundary_C_B)
    print(Boundary_C_B_sum)

    # 计算中心批次和测试批次之间的KECA-DISSIM指标
    X_test1_Slide_KECA = keca_dissim(
        K, k, n2, L, gamma, n_components, -0.1)
    Distances_test1 = np.zeros(n2)
    for i in range(n2):
        Distances_temp = np.linalg.norm(X_test1_Slide_KECA[(K-L+1)*i:((K-L+1)*(i+1)),] -
                                        Center_vector, axis=1)
        Distances_test1[i] = sum(Distances_temp)

    X_test2_Slide_KECA = keca_dissim(
        K, k, n2, L, gamma, n_components, -0.12)
    Distances_test2 = np.zeros(n2)
    for i in range(n2):
        Distances_temp = np.linalg.norm(X_test2_Slide_KECA[(K-L+1)*i:((K-L+1)*(i+1)),] -
                                        Center_vector, axis=1)
        Distances_test2[i] = sum(Distances_temp)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    # ax[0].scatter(range(n2), Distances_test1, color='red')
    # ax[0].set_xlabel('Index')
    # ax[0].set_ylabel('Distances')
    # ax[0].set_title('Test1_-0.6')
    # ax[1].scatter(range(n2), Distances_test2, color='blue')
    # ax[1].set_xlabel('Index')
    # ax[1].set_ylabel('Distances')
    # ax[1].set_title('Test2_-0.9')

    ax[0].plot(range(n2), Distances_test1, color='red')
    ax[0].axhline(y=Boundary_C_B_sum, color='black', linestyle='-')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Distances')
    ax[0].set_title('Test1_-0.1')

    ax[1].plot(range(n2), Distances_test2, color='blue')
    ax[1].axhline(y=Boundary_C_B_sum, color='black', linestyle='-')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Distances')
    ax[1].set_title('Test1_-0.12')

    plt.show()

    End_time = time.time()
    Run_time = End_time - Start_time
    print(Run_time)
