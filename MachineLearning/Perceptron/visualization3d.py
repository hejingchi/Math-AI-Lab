from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 设置字体
from perceptron import *

def plot3d(data, label, random_w, random_b, predict_w, predict_b):
    '''

    :param data: 数据
    :param label: 标签
    :param random_w:
    :param random_b: 原始
    :param predict_w:
    :param predict_b: 预测
    :return: 返回值是是不是二维图像
    '''
    if len(data[0]) !=3:
        print("dimension not equal to 3!")
        return False
    fig = plt.figure() # 创建类
    fig.title("三维平面感知机二分类示意图")
    ax = fig.add_subplot(111, projection="3d")
    N = len(data)

    positive_idx = np.where(label == 1)[0]
    negative_idx = np.where(label == -1)[0]
    x_positive = data[positive_idx, 0] # 第一个坐标
    y_positive = data[positive_idx, 1] # 第二个坐标
    z_positive = data[positive_idx, 2] # 第三个坐标
    ax.scatter(
        x_positive,
        y_positive,
        z_positive
    ) # 正样本
    x_negative = data[negative_idx, 0]
    y_negative = data[negative_idx, 1]
    z_negative = data[negative_idx, 2]
    ax.scatter(
        x_negative,
        y_negative,
        z_negative
    ) # 负样本
    xx, yy = np.meshgrid(np.linspace(-10, 10, N),
                                     np.linspace(-10, 10, N)) # xy坐标上的点
    random_z = (-random_w[0] * xx + random_w[1] * yy + random_b) / random_w[2]
    ax.plot_surface(xx, yy, random_z)
    plt.show()
    return True

def main(): # 仅用于测试代码
    data, label ,random_w, random_b= random_data(100, 3)
    w, b = perceptron(data, label) # 从perceptron中获取参数
    plot3d(data, label, random_w, random_b, w, b)
    return

if __name__ == "__main__":
    main()