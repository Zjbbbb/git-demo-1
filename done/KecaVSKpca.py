import numpy as np
import matplotlib.pyplot as plt

#构建样本数据
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter

X, y = make_circles(n_samples=200, random_state=123,factor=0.3,noise=0.02)

#print(y);

Z , w =make_circles(n_samples=200,factor=0.6,noise=0.02)

plt.scatter(X[y==0, 0], X[y==0, 1], color='r', marker='^', alpha=.4)
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', marker='o', alpha=.4)
plt.scatter(Z[w==1, 0], Z[w==1, 1], color='g', marker='o', alpha=.4)

print(Z)
KY=y
KX=X

count=-1;
for i in w:
    count+=1;
    if i==1:
       w[count]=2;
       KX=np.append(KX,Z[count])
       KY=np.append(KY,w[count])

KX=KX.reshape(int(KX.size/2),2)

print(KY)
print(y);

X=KX
y=KY

#KECA处理结果
def rbf_keca(X, gamma, k):
    #k=8;
    sq_dist = pdist(X, metric='sqeuclidean')
                            # N = X.shape[0]    
                            # sq_dist.shape = N*(N-1)/2
    mat_sq_dist = squareform(sq_dist)
                            # mat_sq_dist.shape = (N, N)
    # step 1
    K = np.exp(-gamma*mat_sq_dist)

    # step 2
    N = X.shape[0]
    one_N = np.ones((N, N))/N
    K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

    # step 3
    Lambda, Q = np.linalg.eig(K)
    eigen_pairs = [(Lambda[i], Q[:, i]) for i in range(len(Lambda))]

    #熵值排序
    Renyi = np.zeros_like(Lambda)
    for i in range(len(Lambda)):
        #Sum_pairs = Q[:, i] * (Lambda[i]**(1/2))
        Renyi[i] = sum(abs( Q[:, i] * (Lambda[i]**(1/2)) ))
    
    Renyi_pairs = [(Renyi[i], Lambda[i], Q[:, i]) for i in range(len(Lambda))]
    #print()
    Renyi_pairs = sorted(Renyi_pairs, reverse=True, key=itemgetter(0))

    New_Q = np.zeros_like(Q)
    for i in range(len(Lambda)):
       New_Q[i]=Renyi_pairs[i][2]
    
    New_Q=New_Q.T
    #还原对角矩阵
    f=np.dot(np.dot(np.linalg.inv(New_Q),K),New_Q)
    f=np.around(f,decimals=8)

    TargetF= np.column_stack((f[:,i] for i in range(k)))
    TargetQ= np.column_stack((New_Q[:,i] for i in range(k)))

    #testF=TargetF**(1/2)

    TargetECA= np.dot(TargetF**(1/2),TargetQ.T)
    TargetECA= np.row_stack((TargetECA[i,:] for i in range(k)))

    #eigen_pairs = sorted(eigen_pairs, reverse=True, key=lambda k: k[0])
    #eigen_pairs[i][1] for i in range(k)
    return TargetECA.T

def rbf_kpca(X, gamma, k):
    sq_dist = pdist(X, metric='sqeuclidean')
                            # N = X.shape[0]    
                            # sq_dist.shape = N*(N-1)/2
    mat_sq_dist = squareform(sq_dist)
                            # mat_sq_dist.shape = (N, N)
    # step 1
    K = np.exp(-gamma*mat_sq_dist)

    # step 2
    N = X.shape[0]
    one_N = np.ones((N, N))/N
    K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

    # step 3
    Lambda, Q = np.linalg.eig(K)
    eigen_pairs = [(Lambda[i], Q[:, i]) for i in range(len(Lambda))]
    eigen_pairs = sorted(eigen_pairs, reverse=True, key=lambda k: k[0])
    return np.column_stack((eigen_pairs[i][1] for i in range(k)))

#调用上述自定义函数
X_keca = rbf_keca(X, gamma=15, k=3)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_keca[y==0, 0], X_keca[y==0, 1],X_keca[y==0, 2], color='r', marker='^', alpha=.4)
ax.scatter(X_keca[y==1, 0], X_keca[y==1, 1],X_keca[y==1, 2], color='b', marker='o', alpha=.4)
ax.scatter(X_keca[y==2, 0], X_keca[y==2, 1],X_keca[y==2, 2], color='g', marker='o', alpha=.4)

#将降维处理后的数据绘制出来
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(X_keca[y==0, 0], X_keca[y==0, 1], color='r', marker='^', alpha=.4)
ax[0].scatter(X_keca[y==1, 0], X_keca[y==1, 1], color='b', marker='o', alpha=.4)
ax[0].scatter(X_keca[y==2, 0], X_keca[y==2, 1], color='g', marker='o', alpha=.4)
label_count = np.bincount(y)
                                # 统计各类别出现的次数
                                # label_count[0] = 500
                                # label_count[1] = 500
ax[1].scatter(X_keca[y==1, 2], np.zeros(label_count[1]), color='b')
ax[1].scatter(X_keca[y==2, 2], np.zeros(label_count[1]), color='g')
ax[1].scatter(X_keca[y==0, 2], np.zeros(label_count[0]), color='r')
                                # y轴置零
                                # 投影到x轴
#ax[2].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],X_kpca[y==0, 2], color='r', marker='^', alpha=.4)
#ax[2].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],X_kpca[y==1, 2], color='b', marker='o', alpha=.4)
#ax[2].scatter(X_kpca[y==2, 0], X_kpca[y==2, 1],X_kpca[y==2, 2], color='g', marker='o', alpha=.4)

ax[1].set_ylim([-1, 1])
ax[0].set_xlabel('EC1')
ax[0].set_ylabel('EC2')
ax[1].set_xlabel('EC1')

#调用上述自定义函数
X_kpca = rbf_kpca(X, gamma=15, k=3)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],X_kpca[y==0, 2], color='r', marker='^', alpha=.4)
ax.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],X_kpca[y==1, 2], color='b', marker='o', alpha=.4)
ax.scatter(X_kpca[y==2, 0], X_kpca[y==2, 1],X_kpca[y==2, 2], color='g', marker='o', alpha=.4)

#将降维处理后的数据绘制出来
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='r', marker='^', alpha=.4)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='b', marker='o', alpha=.4)
ax[0].scatter(X_kpca[y==2, 0], X_kpca[y==2, 1], color='g', marker='o', alpha=.4)
label_count = np.bincount(y)
                                # 统计各类别出现的次数
                                # label_count[0] = 500
                                # label_count[1] = 500
ax[1].scatter(X_kpca[y==1, 2], np.zeros(label_count[1]), color='b')
ax[1].scatter(X_kpca[y==2, 2], np.zeros(label_count[1]), color='g')
ax[1].scatter(X_kpca[y==0, 2], np.zeros(label_count[0]), color='r')
                                # y轴置零
                                # 投影到x轴
#ax[2].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],X_kpca[y==0, 2], color='r', marker='^', alpha=.4)
#ax[2].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],X_kpca[y==1, 2], color='b', marker='o', alpha=.4)
#ax[2].scatter(X_kpca[y==2, 0], X_kpca[y==2, 1],X_kpca[y==2, 2], color='g', marker='o', alpha=.4)

ax[1].set_ylim([-1, 1])
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_xlabel('PC1')

plt.show()