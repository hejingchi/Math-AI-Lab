# Perceptron 感知机

使用 NumPy 实现感知机原始形式与对偶形式。

## 内容

- 原始感知机
- 对偶感知机 
- Gram矩阵 (特殊的核函数)
- 可视化实验 (2d 3d可视化)

## 数学形式

$w^Tx+b=0$

---
# 原理解释
我们这里着重解释一下感知机的对偶算法，$\text{perceptron_dual}$ 中的这一段代码。
对感知机的原始形式，我们考虑使用梯度下降法，每次梯度更新采用

    for epoch in range(5):
        # epoch是执行轮数
        running_loss = 0 # 损失函数
        for i in range(N): # 把每一个样本都遍历一遍
            if label[i] * (data[i] @ w + b) <= 0: # 如果分类有问题
                running_loss += 1 # 损失函数
                w += lr * label[i] * data[i]
                b += lr * label[i] # 梯度下降法

通过这个方式更新权重和偏置。但其实我们会发现，$\text{w}$ 和 $\text{b}$ 本质上是再最开始用随机数初始化之后，
用随机梯度下降法，每次沿着梯度的方向，走过 $\text{lr}$ 这样一段长度去进行一个逼近。
其中，我们考虑把 $\text{w}$ 具体写下来：
    
    # 我们不妨假设前k个是刚好需要做梯度处理的。也就是说，他们满足这样的关系：
    # label[i] * (data[i] @ w + b) <= 0
    #于是就有
    w_init = rng.uniform(-1, 1, dim) # 初始化
    w_0 = w_init + lr * label[0] * data[0] # 0
    w_1 = w_0 + lr * label[1] * data[1] # 1
    ...
    w_k = w_{k-1} + lr * label[k] * data[k] # k
    
可以发现，这其实就是对内积的一个累加的过程。我们类似地可以看一下对 $\text{b}$ 是怎么处理的。

    b_init = rng.uniform(-1, 1) 
    b_0 = b_init + lr * label[0]
    b_1 = b_0 + lr * label[1]
    ...
    b_k = b_{k-1} + lr * label[k] # 注意这里的label[i]是取±1的

于是我们就看到了，这是一个很想简单的线性递推。因此

    idx = [:k]
    w = lr * label[idx] * data[idx, :] # 这里就是对梯度的处理
    b = lr * np.sum(label[idx])  # 相当于是正部减去负部

也就是说，我们做了一轮 $\text{epoch}$ 之后，处理了一批 $\text{data}$。那我们换个角度想，我们其实可以在最初就把
这个 $\text{idx}$ 给算出来的。这就是感知机的对偶方法。不过我们在这里参数就要换了。
我们把对某一个 $\text{data[i]}$ 进行的累加次数作为我们的参数，也就是这里的 $\text{alpha}$。具体是怎么做的呢？
我们下面来看一下
---
首先我们考虑用 $\text{alpha}$ 作为参数，它是一个 $\text{N}$ 维向量。然后最终的表达式结果中，我们能看到

$$ w = \sum_{i=1}^{N} \alpha_i y_i  x_i,b=\sum_{i=1}^{N} \alpha_i  y_i  $$
是我们最终的结果。注意这里的 $\alpha_i$ 实际上是由两部分组成的，第一部分是 $\text{lr}$，第二部分才是那个累加次数。
但是我们这里把两者结合起来。在 $\text{SVM}$ 中我们将会看到这种方法的好处。不过暂时不讲这个。

于是我们惊奇的发现，仅仅是用这个角度取看待 $\text{w}$ 之后，我们如果考虑某一个数据 $x_j$，那么

$$ w \cdot x_j + b =  \left(\sum_{i=1}^{N} \alpha_i y_i \left( x_i\cdot x_j \right)\right)+b$$

其中熟知 $\text{gram}$ 矩阵 $$ \text{gram}_{i,j}=x_i^Tx_j$$并且后文在不引起歧义的情况下我们都省略转置符号，
直接用向量相乘表示两个向量的内积。因此我们就相当于得到了 $x_j$ 的分类结果

为了得到全体数据的分类结果，我们只需要

$$ w \cdot \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_N \end{bmatrix} = 
\begin{bmatrix} w \cdot x_1 \\ w \cdot x_2 \\ \vdots \\ w \cdot x_N \end{bmatrix} 
= \begin{bmatrix} \sum_{i=1}^{N} \alpha_i y_i (x_i \cdot x_1) \\ 
\sum_{i=1}^{N} \alpha_i y_i (x_i \cdot x_2) \\ 
\vdots \\ 
\sum_{i=1}^{N} \alpha_i y_i (x_i \cdot x_N) \end{bmatrix} $$

加上偏置 $b$ 后，全体样本的决策函数值为：

$$ \begin{bmatrix} w \cdot x_1 + b \\ w \cdot x_2 + b \\ \vdots \\ w \cdot x_N + b \end{bmatrix} 
= \begin{bmatrix} \sum_{i=1}^{N} \alpha_i y_i (x_i \cdot x_1) \\ 
\sum_{i=1}^{N} \alpha_i y_i (x_i \cdot x_2) \\ 
\vdots \\ 
\sum_{i=1}^{N} \alpha_i y_i (x_i \cdot x_N) \end{bmatrix} + b \cdot \mathbf{1} $$

其中 $\mathbf{1}$ 为 $N$ 维全1列向量。

注意到 $\sum_{i=1}^{N} \alpha_i y_i (x_i \cdot x_j)$ 正是 $\text{gram}$ 矩阵第 $j$ 列

与向量 $[\alpha_1 y_1, \dots, \alpha_N y_N]^T$ 的内积。则

$$ \boxed{ \begin{bmatrix} w \cdot x_1 + b \\ w \cdot x_2 + b \\ \vdots \\ w
\cdot x_N + b \end{bmatrix} = \text{gram} \cdot y\cdot \alpha + b \cdot \mathbf{1} } $$



    for epoch in range(5):
        running_loss = 0
        for i in range(N): # 进行N次输出一下误差函数
            margin = label * (gram @ (label * alpha) + b) # 计算有问题的点
            idx = np.where(margin <= 0)[0] # 找出所有错分类的下标
            alpha[idx] += lr # 对alpha更新其学习率
            b += lr * np.sum(label[idx]) # 梯度更新 一次性更新一批 而且是要根据方向去更新
            running_loss = len(idx) / N

这是感知机对偶方法的的核心代码。
我们首先引入了 $\text{gram}$ 矩阵，处理了 $\text{data}$ 函数之间的内积。

这其实就是给出了一个核函数，所谓核函数就是满足$K(x, z) = \Phi(x) \Phi(z) $的函数 $K$，此外还有一些条件例如半正定等等.
高等代数中的 $\text{gram}$ 矩阵就是一个矩阵的转置和其乘积。这一方面可以看成矩阵的列向量的两两内积，另一方面，由于

$$ x^TA^TAx=(Ax)^T(Ax) \ge 0$$ 

其实就给出了 $\text{gram}$ 矩阵半正定的证明。
其实就给出了 $\text{gram}$ 矩阵半正定的证明。这一个是核函数的要求之一