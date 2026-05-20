import numpy as np

def random_data_hard(N=100, dim=2):
    '''
    根据输入的参数或者默认参数生成一组数据，包括样本数据和样本类型
    生成的是线性可分的数据集
    :param N: 数据数量
    :param dim: 数据维数
    :return: data: 数据
    :return: y: 标签
    :return: w: 随机超平面的法向量
    :return: b: 随机超平面的偏置
    '''
    rng = np.random.default_rng(seed=42)
    data = rng.uniform(-10, 10, (N, dim))
    w = rng.uniform(-1, 1, dim) # 超平面
    b = rng.uniform(-1, 1) # 偏置
    y = (data @ w + b > 0) * 2 - 1 # 标签 numpy
    return data, y, w, b

def random_data_soft(N=10000, dim=2):
    '''
    根据输入的参数或者默认参数生成一组数据，包括样本数据和样本类型
    生成的是线性不可分的，需要引入松弛
    :param N: 数据数量
    :param dim: 数据维数
    :return: data: 数据
    :return: y: 标签
    :return: w: 随机超平面的法向量
    :return: b: 随机超平面的偏置
    '''
    rng = np.random.default_rng()
    data = rng.uniform(-10, 10, (N, dim))
    w = rng.uniform(-1, 1, dim) # 超平面
    b = rng.uniform(-1, 1) # 偏置
    y = (data @ w + b > 0) * 2 - 1 # 标签 numpy
    print(np.sum(y))
    error = rng.random(size = N)
    p = 0.05# 随机给一些点，让他们的编号恰好不对
    error_idx = np.where(error > 1 - p)[0] # 获取这些点的编号
    y[error_idx] = -y[error_idx]
    print(np.sum(y))
    return data, y, w, b

def svm_hard(data, y):
    '''

    :param data: 数据集
    :param y: 数据标签
    :return: w: svm的超平面法向量
    :return: b: svm的偏置
    '''
    N = len(data) # 数据数量
    dim = len(data[0]) # 维度
    lr = 0.001 # 学习率设置
    rng = np.random.default_rng()
    running_loss = 0 # 损失函数
    w = np.zeros(N) # 初始化原始问题的解
    b = 0 # 初始化标量
    alpha = np.zeros(N) # 初始化对偶问题的解
    # \alpha 为参数
    # 对对偶问题，我们需要求解以\alpha为自变量的目标函数
    # 并最终利用KKT条件得到关于 w 与 b的式子
    # 目标函数 f(\alpha) = \frac{1}{2} \alpha^T \cdot gram \alpha - \alpha
    gram = y.T @ data @ data.T @ y # 计算gram矩阵，本质上是核函数
    print(y.T)
    print(data)
    print(gram)
    for epoch in range(5):
        running_loss = 0
        for i in range(N):
            return

    print(f"linear svm: {np.array2string(w, precision=2)} @ x + {b:.4f}")
    return w, b

def svm_soft(data, y):
    '''
    软间隔支撑向量机 linear
    :param data:
    :param y:
    :return:
    '''
    N = len(data)
    dim = len(data[0])
    rng = np.random.default_rng()
    w = rng.uniform(-1, 1, N)
    b = rng.uniform(-1, 1)


    return w, b

def main():
    data1, y1, random_w1, random_b1 = random_data_hard(10, 2) # 硬间隔
    #data2, y2, random_w2, random_b2 = random_data_soft() # 两个数据初始化
    print("="*30)
    print("="*10, "linear svm", "="*10)
    svm_hard(data1, y1) # 硬间隔
    print()
    '''
    print("="*30)
    print("="*10, "unlinear svm", "="*10)
    svm_soft(data2, y2)
    print()
    return
    '''

if __name__ == "__main__":
    main()