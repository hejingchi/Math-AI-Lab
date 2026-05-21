from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from MachineLearning.Perceptron.perceptron import *

def plot2d(data, label, random_w, random_b, predict_w, predict_b):
    '''

    :param data: 数据
    :param label: 标签
    :param random_w:
    :param random_b: 原始
    :param predict_w:
    :param predict_b: 预测
    :return: 返回值是是不是二维图像
    '''
    if len(data[0]) !=2:
        print("dimension not equal to 2!")
        return False
    plt.figure("二维平面感知机二分类示意图")
    N = len(data)
    #xlim = np.linspace(-10, 10, 1)
    #ylim = np.linspace(-10, 10, 1)
    positive_idx = np.where(label == 1)[0]
    negative_idx = np.where(label == -1)[0]
    x_positive = data[positive_idx, 0] # 第一个坐标
    y_positive = data[positive_idx, 1] # 第二个坐标
    x_negative = data[negative_idx, 0]
    y_negative = data[negative_idx, 1]
    plt.scatter(x_positive, y_positive, c='b')
    plt.scatter(x_negative, y_negative, c='r')
    plt.xlabel("x轴")
    plt.ylabel("y轴") # 坐标
    #plt.legend()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10) # 左坐标轴
    xline = np.linspace(-10, 10, N)

    random_yline = (
            -(random_w[0] * xline + random_b)
            / random_w[1]
    )
    plt.plot(xline, random_yline) # 原始直线

    positive_yline = (
            -(predict_w[0] * xline + predict_b + 1)
            / predict_w[1]
    ) #
    plt.plot(xline, positive_yline, 'k--')

    netative_yline = (
            -(predict_w[0] * xline + predict_b - 1)
            / predict_w[1]
    ) #
    # 在你的 plot2d 里增加这一段，不加减 1
    main_predict_yline = (-(predict_w[0] * xline + predict_b) / predict_w[1])
    plt.plot(xline, main_predict_yline, 'g-', label='预测主决策面')

    plt.plot(xline, netative_yline, 'k--')
    plt.show()
    return True

def main(): # 仅用于测试代码
    data, label ,random_w, random_b= random_data(100, 2)
    w, b = perceptron(data, label) # 从perceptron中获取参数
    plot2d(data, label, random_w, random_b, w, b)
    return

if __name__ == "__main__":
    main()