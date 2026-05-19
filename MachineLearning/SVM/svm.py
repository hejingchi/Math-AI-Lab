import numpy as np

def random_data_hard(N=100, dim=2):
    '''
    根据输入的参数或者默认参数生成一组数据，包括样本数据和样本类型
    生成的是线性可分的数据集
    :param N: 数据数量
    :param dim: 数据维数
    :return: data: 数据
    :return: label: 标签
    :return: w: 随机超平面的法向量
    :return: b: 随机超平面的偏置
    '''
    rng = np.random.default_rng(seed=42)
    data = rng.uniform(-10, 10, (N, dim))
    w = rng.uniform(-1, 1, dim) # 超平面
    b = rng.uniform(-1, 1) # 偏置
    label = (data @ w + b > 0) * 2 - 1 # 标签 numpy
    return data, label, w, b

def random_data_soft(N=10000, dim=2):
    '''
    根据输入的参数或者默认参数生成一组数据，包括样本数据和样本类型
    生成的是线性不可分的，需要引入松弛
    :param N: 数据数量
    :param dim: 数据维数
    :return: data: 数据
    :return: label: 标签
    :return: w: 随机超平面的法向量
    :return: b: 随机超平面的偏置
    '''
    rng = np.random.default_rng(seed=42)
    data = rng.uniform(-10, 10, (N, dim))
    w = rng.uniform(-1, 1, dim) # 超平面
    b = rng.uniform(-1, 1) # 偏置
    label = (data @ w + b > 0) * 2 - 1 # 标签 numpy
    error = rng.random(size = N)
    p = 0.05 # 随机给一些点，让他们的编号恰好不对
    error_idx = np.where(error > p)[0] # 获取这些点的编号
    right_label = label
    label[error_idx] = -label[error_idx]
    print(label[error_idx])
    print(np.sum(rng.random(size = N) > 0.05))
    return data, label, w, b

def svm_hard(data, label):
    '''

    :param data: 数据集
    :param label: 数据标签
    :return: w: svm的超平面法向量
    :return: b: svm的偏置
    '''
    N = len(data) # 数据数量
    dim = len(data[0]) # 维度
    lr = 0.001 # 学习率设置
    rng = np.random.default_rng()
    w = rng.uniform(-1, 1, dim)
    b = rng.uniform(-1, 1) # 目标输出
    running_loss = 0 # 损失函数
    for epoch in range(5):
        running_loss = 0
        for i in range(N):
            return






    print(f"linear svm: {np.array2string(w, precision=2)} @ x + {b:.4f}")
    return w, b

def svm_soft(data, label):
    '''
    软间隔支撑向量机 linear
    :param data:
    :param label:
    :return:
    '''
    N = len(data)
    dim = len(data[0])
    rng = np.random.default_rng()
    w = rng.random(-1, 1, N)
    b = rng.random(-1, 1)


    return w, b

def main():
    data1, label1, random_w1, random_b1 = random_data_hard() # 硬间隔
    data2, label2, random_w2, random_b2 = random_data_soft() # 两个数据初始化
    print("="*30)
    print("="*10, "linear svm", "="*10)
    svm_hard(data1, label1)
    print("="*30)
    print("="*10, "unlinear svm", "="*10)
    svm_soft(data2, label2)
    return

if __name__ == "__main__":
    main()