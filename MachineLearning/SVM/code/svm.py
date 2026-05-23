import numpy as np
from MachineLearning.SVM.svm_visualization import plot2d
eps = 1e-6

def random_data_hard(N=100, dim=2, margin = 1):
    '''
    根据输入的参数或者默认参数生成一组数据，包括样本数据和样本类型
    生成的是线性可分的数据集
    我们生成一个具有明显支撑向量性质的数据集
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
    w_unit = w / np.linalg.norm(w)
    # 正负类分别平移
    data = data + margin * y[:, None] * w_unit
    return data, y, w, b

def random_data_soft(N=10000, dim=2, margin = 1):
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
    w_unit = w / np.linalg.norm(w)
    # 正负类分别平移
    data = data + margin * y[:, None] * w_unit
    print(np.sum(y))
    return data, y, w, b

def svm_hard(x, y):
    '''

    :param data: 数据集
    :param y: 数据标签
    :return: w: svm的超平面法向量
    :return: b: svm的偏置
    '''
    N = len(x) # 数据数量
    dim = len(x[0]) # 维度
    lr = 1e-6 # 学习率设置
    rng = np.random.default_rng()
    running_loss = 0 # 损失函数
    w = np.zeros(dim) # 初始化原始问题的解
    b = 0 # 初始化标量
    rng = np.random.default_rng(seed=42)
    alpha = rng.uniform(0, 1, N) # 初始化对偶问题的解
    # \alpha 为参数
    # 对对偶问题，我们需要求解以\alpha为自变量的目标函数
    # 并最终利用KKT条件得到关于 w 与 b的式子
    # 目标函数 f(\alpha) =
    # \frac{1}{2} \sum \alpha_i\alpha_j
    # y_i y_j (x_i \cdot x_j)
    gram_x = np.array(x @ x.T) # 计算向量内积
    gram_y = np.array(y[:, None] @ y[None, :])
    G = gram_y * gram_x
    # data是N*2的 gram阵是N*N的
    # y[:,None]是N*1的标签列向量 y[None,:]是1*N的标签行向量
    # 我们计算两个矩阵的逐个元素的乘积。
    # 于是目标函数变为了 f(\alpha) = \frac{1}{2} \sum_{i, j} \alpha_i \alpha j gram[i][j]
    # 我们极小化这个函数肯定要算梯度 这个本质上是二次型
    # f(\alpha) = \frac{1}{2} \alpha^T gram \alpha 其中gram是对称正定矩阵
    for epoch in range(1000):
        running_loss = 0
        # lr = lr / (1.0 + 0.001 * epoch)
        for i in range(N * 10):
            grad_f = G @ alpha - 1 # 计算梯度
            alpha_new = alpha - lr * grad_f # 处理梯度
            alpha_project = alpha_new - y * (y @ alpha_new) / (y @ y) # 投影
            alpha_positive = np.maximum(alpha_project, 0) # 投影到0
            alpha = alpha_positive #
            print("约束偏差 (sum of alpha * y):", np.sum(alpha * y))
        w = np.sum(alpha[:, None] * y[:, None] * x, axis = 0) # 计算w
        positive_idx = np.where(alpha > 0)[0] # 找到第一个大于0的下表
        j = positive_idx[0]
        b = y[j] - w @ x[j] # 计算b
        correct_idx = np.where(y * (x @ w + b) > 0)[0]
        correct_len = len(correct_idx)
        running_loss = (N - correct_len) / N

        print(f"running_loss = {running_loss:.3f}")

    print(f"linear svm: {np.array2string(w, precision=2)} @ x + {b:.4f}")
    return w, b

def svm_hard_test(x, y):
    '''
    上面那个代码是再处理梯度之后，先投影到超平面再去0
    我们在这个算法里面先去0再投影
    :param data: 数据集
    :param y: 数据标签
    :return: w: svm的超平面法向量
    :return: b: svm的偏置
    '''
    N = len(x) # 数据数量
    dim = len(x[0]) # 维度
    lr = 1e-4# 学习率设置
    rng = np.random.default_rng()
    running_loss = 0 # 损失函数
    w = np.zeros(dim) # 初始化原始问题的解
    b = 0 # 初始化标量
    rng = np.random.default_rng()
    alpha = rng.uniform(0, 1, N) # 初始化对偶问题的解
    # \alpha 为参数
    # 对对偶问题，我们需要求解以\alpha为自变量的目标函数
    # 并最终利用KKT条件得到关于 w 与 b的式子
    # 目标函数 f(\alpha) =
    # \frac{1}{2} \sum \alpha_i\alpha_j
    # y_i y_j (x_i \cdot x_j)
    gram_x = np.array(x @ x.T) # 计算向量内积
    gram_y = np.array(y[:, None] @ y[None, :])
    G = gram_y * gram_x
    # data是N*2的 gram阵是N*N的
    # y[:,None]是N*1的标签列向量 y[None,:]是1*N的标签行向量
    # 我们计算两个矩阵的逐个元素的乘积。
    # 于是目标函数变为了 f(\alpha) = \frac{1}{2} \sum_{i, j} \alpha_i \alpha j gram[i][j]
    # 我们极小化这个函数肯定要算梯度 这个本质上是二次型
    # f(\alpha) = \frac{1}{2} \alpha^T gram \alpha 其中gram是对称正定矩阵
    for epoch in range(100):
        running_loss = 0
        # lr = lr / (1.0 + 0.001 * epoch)
        for i in range(N * 10):
            grad_f = G @ alpha - 1 # 计算梯度
            alpha_new = alpha - lr * grad_f # 处理梯度
            alpha_positive = np.maximum(alpha_new, 0)
            alpha_project = alpha_positive - y * (y @ alpha_positive) / (y @ y) # 投影到超平面
            alpha = alpha_project #
        w = np.sum(alpha[:, None] * y[:, None] * x, axis = 0) # 计算w
        positive_idx = np.where(alpha > 0)[0] # 找到第一个大于0的下表
        j = positive_idx[0]
        b = y[j] - w @ x[j] # 计算b
        correct_idx = np.where(y * (x @ w + b) > 0)[0]
        correct_len = len(correct_idx)
        running_loss = (N - correct_len) / N

        print(f"running_loss = {running_loss:.3f}")

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

    data1, y1, random_w1, random_b1 = random_data_hard(100, 2) # 硬间隔
    #data2, y2, random_w2, random_b2 = random_data_soft() # 两个数据初始化
    print("="*30)
    print("="*10, "linear svm", "="*10)
    predict_w1, predict_b1 = svm_hard(data1, y1) # 硬间隔
    print(f"random data is {np.array2string(random_w1,precision=2)} @ x + {random_b1:.2f}")
    plot2d(data1, y1, random_w1, random_b1, predict_w1, predict_b1)
    '''
    print("="*30)
    print("="*10, "unlinear svm", "="*10)
    svm_soft(data2, y2)
    print()
    return
    '''

if __name__ == "__main__":
    main()