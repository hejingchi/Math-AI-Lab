import numpy as np

def random_data(N=100, dim=2):
    '''

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

def svm(data, label):
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
    return w, b

def main():
    data, label, w, b = random_data()
    print("="*30)
    print("="*10, "svm", "="*10)
    svm(data, label)
    return

if __name__ == "__main__":
    main()