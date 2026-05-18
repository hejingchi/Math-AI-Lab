import numpy as np

def random_data(N = 5, dim = 2):
    '''
    :peremeter:
    N: 数据的个数 默认值100
    dim: 数据的维数 默认值2
    为线性可分感知机模型创建一个数据集
    创建一个二维空间中的数据集，采用直线 wx+b=0 进行划分，w和b使用numpy随机数进行生成
    再随机生成n个数据点data[i] = [x_i, y_i]，
    x = data[i]，若wx+b>0则把该数据点划分为正类，label[i] = 1
    否则是负类，label[i] = 0
    :return:
    data 数据集，包括n个数对(x_i, y_i), i=1, 2,..., n
    label 数据标签 表示数据是正类还是负类
    '''
    rng = np.random.default_rng(seed = 128) #随机种子
    w = rng.uniform(-1, 1, dim) # 创建一维数组
    b = rng.uniform(-1, 1) # 创建偏置
    data = rng.uniform(-10, 10, (N, dim)) # 数据集创建 100*2
    label = (data @ w + b > 0) * 2 - 1 # 分类类别设置
    label = np.array(label) # 转化为numpy 防止出现问题
    print(f"random plane {np.array2string(w, precision=2)} @ x + {b:.2f} = 0")
    return data, label, w, b

def perceptron(data, label):
    '''

    :param data: 数据集
    :param label: 类别
    :return: 返回值是w和b
    '''
    N = len(data) # 数据集长度
    dim = len(data[0]) # 数据维数

    lr = 0.001 # 学习率
    rng = np.random.default_rng()
    w = rng.uniform(-1, 1, dim) # 一维数组
    b = rng.uniform(-1, 1) # 随机数初始化

    running_loss = 0 # 损失函数
    for epoch in range(5):
        # epoch是执行轮数
        running_loss = 0
        for i in range(N):
            if label[i] * (data[i] @ w + b) <= 0: # 如果分类有问题
                running_loss += 1 # 损失函数
                w += lr * label[i] * data[i]
                b += lr * label[i] # 梯度下降法
        print(f"epoch {epoch}, running_loss {running_loss / N:.3f}")
    print(f"predicting plane {np.array2string(w,precision=2)} @ x + {b:.2f} = 0")
    return w, b
    # 注意事项：numpy数组不能直接用格式化输出，必须用np.array2string(<数组>, precision=p)表示精确到p位小数
    # 而如果知识简单的标量，就不用这么输出
    # b = rng.uniform(-1, 1)是标量
    # b = rng.uniform(-1, 1, (1, 1))是1*1的numpy矩阵
    # b = rng.uniform(-1, 1, (1,))是1维的一维数组
    # w = rng.uniform(-1, 1, (dim, 1))是dim*1的numpy矩阵
    # w = rng.uniform(-1, 1, (dim, 1))是dim维的一维数组

def perceptron_dual(data, label):
    '''

    :param data: 数据集
    :param label: 类别
    :return: 返回w和b的值
    '''
    N = len(data) # 数据集个数
    dim = len(data[0]) # 维度
    lr = 0.0001 # 学习率
    rng = np.random.default_rng()
    alpha = np.zeros(N) # 对偶方法
    b = rng.uniform(-1, 1) # 随机数初始化
    gram = data @ data.T # 提前计算gram矩阵
    running_loss = 0 # 损失函数
    for epoch in range(5):
        running_loss = 0
        for i in range(N): # 进行N次输出一下误差函数
            margin = label * (gram @ (label * alpha) + b) # 计算有问题的点
            idx = np.where(margin <= 0)[0] # 找出所有错分类的下标
            alpha[idx] += lr # 对alpha更新其学习率
            b += lr * np.sum(label[idx]) # 梯度更新 一次性更新一批 而且是要根据方向去更新
            running_loss = len(idx) / N
        print(f"epoch {epoch}, running_loss {running_loss:.3f}")
    w = alpha * label @ data
    print(f"predicting dual plane {np.array2string(w,precision=2)} @ x + {b:.2f} = 0")
    return w, b

def main():
    data, label ,random_w, random_b= random_data(100, 3)

    print("="*10, "perceptron", "="*10)
    w, b = perceptron(data, label)
    direction_loss = w @ random_w / np.linalg.norm(random_w) / np.linalg.norm(w)
    print(f"direction_loss {direction_loss:.4f}") # 方向差
    print("=" * 30)

    w2, b2 = perceptron_dual(data, label)
    direction_loss_dual = w2 @ random_w / np.linalg.norm(random_w) / np.linalg.norm(w2)
    print(f"direction_loss_dual {direction_loss_dual:.4f}")
    print("=" * 30)

if __name__ == "__main__":
    main()
