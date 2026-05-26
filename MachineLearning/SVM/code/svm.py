import numpy as np
from svm_visualization import plot2d
eps = 1e-6
C = 1

def learning_rate(x, safety_factor = 0.5):
    if safety_factor > 1:
        print("安全系数超过1！")
        return False
    M = np.linalg.norm(x, 2) # 计算Frobenius范数
    lr = 2.0 / M # 学习率收敛最大值
    lr *= safety_factor # 安全系数
    return lr

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

def random_data_soft(N=100, dim=2, margin = 1):
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
    error = rng.random(size = N)
    p = 0.1 # 随机给一些点，让他们的编号恰好不对
    error_idx = np.where(error > 1 - p)[0] # 获取这些点的编号
    y[error_idx] = -y[error_idx]
    w_unit = w / np.linalg.norm(w)
    # 正负类分别平移
    data = data + margin * y[:, None] * w_unit
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
    lr = learning_rate(G)
    # data是N*2的 gram阵是N*N的
    # y[:,None]是N*1的标签列向量 y[None,:]是1*N的标签行向量
    # 我们计算两个矩阵的逐个元素的乘积。
    # 于是目标函数变为了 f(\alpha) = \frac{1}{2} \sum_{i, j} \alpha_i \alpha j gram[i][j]
    # 我们极小化这个函数肯定要算梯度 这个本质上是二次型
    # f(\alpha) = \frac{1}{2} \alpha^T gram \alpha 其中gram是对称正定矩阵
    for epoch in range(20):
        running_loss = 0
        # lr = lr / (1.0 + 0.001 * epoch)
        for i in range(N * 10):
            grad_f = G @ alpha - 1 # 计算梯度
            alpha_new = alpha - lr * grad_f # 处理梯度
            alpha_project = alpha_new - y * (y @ alpha_new) / (y @ y) # 投影
            alpha_positive = np.maximum(alpha_project, 0) # 投影到0
            alpha = alpha_positive #
        w = np.sum(alpha[:, None] * y[:, None] * x, axis = 0) # 计算w
        positive_idx = np.where(alpha > 0)[0] # 找到第一个大于0的下表
        j = positive_idx[0]
        b = y[j] - w @ x[j] # 计算b
        correct_idx = np.where(y * (x @ w + b) > 0)[0]
        correct_len = len(correct_idx)
        f_alpha = 0.5 * alpha @ G @ alpha - np.sum(alpha) # 实际上在算的损失函数
        if epoch % 10 == 0:
            print(f"f_alpha = {f_alpha:.3f}")

    print(f"linear svm: {np.array2string(w, precision=2)} @ x + {b:.4f}")
    return w, b

def svm_soft(x, y):
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
    lr = learning_rate(G, 1)
    print(f"lr = {lr:.2e}")
    # data是N*2的 gram阵是N*N的
    # y[:,None]是N*1的标签列向量 y[None,:]是1*N的标签行向量
    # 我们计算两个矩阵的逐个元素的乘积。
    # 于是目标函数变为了 f(\alpha) = \frac{1}{2} \sum_{i, j} \alpha_i \alpha j gram[i][j]
    # 我们极小化这个函数肯定要算梯度 这个本质上是二次型
    # f(\alpha) = \frac{1}{2} \alpha^T gram \alpha 其中gram是对称正定矩阵
    for epoch in range(20):
        running_loss = 0
        # lr = lr / (1.0 + 0.001 * epoch)
        for i in range(N * 10):
            grad_f = G @ alpha - 1 # 计算梯度
            alpha_new = alpha - lr * grad_f # 处理梯度
            alpha_project = alpha_new - y * (y @ alpha_new) / (y @ y) # 投影
            alpha_positive = np.maximum(alpha_project, 0) #
            alpha_positive = np.minimum(alpha_positive, C) # 投影到 0 < alpha < C
            alpha = alpha_positive # 计算更新之后的结果
        w = np.sum(alpha[:, None] * y[:, None] * x, axis = 0) # 计算w
        positive_idx = np.where(alpha > 0)[0] # 找到第一个大于0的下表
        j = positive_idx[0]
        sv_idx = np.where((alpha > 1e-6) & (alpha < C - 1e-6))[0]
        if len(sv_idx) > 0:
            j = sv_idx[0]
            b = y[j] - w @ x[j]
        else:
            # 如果找不到严格在 (0, C) 内的点，退而求其次用 alpha > 0 的点
            positive_idx = np.where(alpha > 1e-6)[0]
            j = positive_idx[0]
            b = y[j] - w @ x[j]
        correct_idx = np.where(y * (x @ w + b) > 0)[0]
        correct_len = len(correct_idx)
        accuracy = correct_len / N
        f_alpha = 0.5 * alpha @ G @ alpha - np.sum(alpha)
        if epoch % 10 == 9:
            print(f"epoch = {epoch + 1}, f_alpha = {f_alpha:.3f}, accuracy = {accuracy*100:.2f}%")
            # 输出轮次、误差和准确度

    print(f"linear svm: {np.array2string(w, precision=2)} @ x + {b:.4f}")
    return w, b

def svm_smo(x, y):
    '''

       :param x: 数据集
       :param y: 数据标签
       :return: w: svm的超平面法向量
       :return: b: svm的偏置
       '''
    N = len(x)  # 数据数量
    dim = len(x[0])  # 维度
    lr = 1e-6  # 学习率设置
    rng = np.random.default_rng()
    running_loss = 0  # 损失函数
    w = np.zeros(dim)  # 初始化原始问题的解
    b = 0  # 初始化标量
    rng = np.random.default_rng(seed=42)
    alpha = rng.uniform(0, 1, N)  # 初始化对偶问题的解

    gram_x = np.array(x @ x.T)  # 计算向量内积
    gram_y = np.array(y[:, None] @ y[None, :])
    G = gram_y * gram_x
    lr = learning_rate(G, 1)
    print(f"lr = {lr:.2e}") # 初始化

    for epoch in range(200):
        running_loss = 0
        for _ in range(N * 10):
            # 初始 full_sum
            full_sum = G @ (alpha * y) # 提前算好求和部分
            # 随机选 i 和 j
            i = rng.integers(0, N)
            j = i
            while j == i:
                j = rng.integers(0, N)
            # 计算 sum_term
            sum_term = ((full_sum[i] - alpha[i] * y[i] * G[i, i] - alpha[j] * y[j] * G[j, i])
                        - (full_sum[j] - alpha[i] * y[i] * G[i, j] - alpha[j] * y[j] * G[j, j])) # 求和的部分提前减去
            # zeta 和 eta
            zeta = y[i] * alpha[i] + y[j] * alpha[j] # 提前算好
            eta = G[i, i] + G[j, j] - 2 * G[i, j] # 分母部分
            # 无约束最优解
            aj_new_unc = (1 - y[i] * y[j] + y[j] * zeta * (G[i, i] - G[i, j]) + y[j] * sum_term) / eta
            # 上下界和截断
            if y[i] != y[j]:
                L = max(0, alpha[j] - alpha[i])
                H = min(C, C + alpha[j] - alpha[i])
            else:
                L = max(0, alpha[i] + alpha[j] - C)
                H = min(C, alpha[i] + alpha[j])
            alpha_j_new = np.clip(aj_new_unc, L, H)
            alpha_i_new = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j_new)
            # 更新 full_sum
            full_sum += y[i] * (alpha_i_new - alpha[i]) * G[i, :] # 只需更新alpha_i 的部分
            full_sum += y[j] * (alpha_j_new - alpha[j]) * G[j, :]
            # 更新 alpha
            alpha[i], alpha[j] = alpha_i_new, alpha_j_new
        w = np.sum(alpha[:, None] * y[:, None] * x, axis=0)  # 计算w
        positive_idx = np.where(alpha > 0)[0]  # 找到第一个大于0的下表
        j = positive_idx[0]
        sv_idx = np.where((alpha > 1e-6) & (alpha < C - 1e-6))[0]
        if len(sv_idx) > 0:
            j = sv_idx[0]
            b = y[j] - w @ x[j]
        else:
            # 如果找不到严格在 (0, C) 内的点，退而求其次用 alpha > 0 的点
            positive_idx = np.where(alpha > 1e-6)[0]
            j = positive_idx[0]
            b = y[j] - w @ x[j]
        correct_idx = np.where(y * (x @ w + b) > 0)[0]
        correct_len = len(correct_idx)
        accuracy = correct_len / N
        f_alpha = 0.5 * alpha @ G @ alpha - np.sum(alpha)
        if epoch % 10 == 9:
            print(f"epoch = {epoch + 1}, f_alpha = {f_alpha:.3f}, accuracy = {accuracy * 100:.2f}%")
    return w, b

def linear():
    data1, y1, random_w1, random_b1 = random_data_hard(100, 10) # 硬间隔
    print("="*10, "linear svm", "="*10)
    predict_w1, predict_b1 = svm_hard(data1, y1) # 硬间隔
    print(f"random data is {np.array2string(random_w1,precision=2)} @ x + {random_b1:.2f}")
    plot2d(data1, y1, random_w1, random_b1, predict_w1, predict_b1)
    print("="*30)

def unlinear():
    data2, y2, random_w2, random_b2 = random_data_soft(100, 10)  # 两个数据初始化
    print("=" * 10, "unlinear svm", "=" * 10)
    predict_w2, predict_b2 = svm_soft(data2, y2)
    plot2d(data2, y2, random_w2, random_b2, predict_w2, predict_b2)
    print("=" * 30)

def smo():
    data3, y3, random_w3, random_b3 = random_data_soft(100, 2) # 数据初始化
    print("=" * 10, "smo", "=" * 10)
    predict_w3, predict_b3 = svm_smo(data3, y3)
    plot2d(data3, y3, random_w3, random_b3, predict_w3, predict_b3)
    print("=" * 30)

def main():

    # linear()
    while 1:
        # linear()
        # unlinear()
        smo()
if __name__ == "__main__":
    main()