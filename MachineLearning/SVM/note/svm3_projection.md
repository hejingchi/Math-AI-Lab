# 从对偶问题回到原问题——利用梯度的投影方法
对最优化问题
$$
\begin{aligned}
\min_\alpha\quad & \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}
\alpha_i\alpha_j y_i y_j \left( x_i\cdot x_j\right) -\sum_{i=1}^{N} \alpha_i\\
\text{s.t.} \quad & \sum_{i=1}^{N} \alpha_i y_i =0  \\
& \alpha_i \geqslant 0
\end{aligned}
$$

也被我们称为是硬间隔的问题，我们需要求解以 $\alpha$ 为自变量的目标函数
并最终利用 $\text{KKT}$ 条件得到关于 $\text{w}$ 与 $\text{b}$ 的式子。前文
我们只讨论了对偶问题的解，并没有仔细说如何把对偶问题的解转化为原问题的解。这篇我们需要做
两件事情，一是如何类比感知机的方法，利用梯度下降法的思想求解原始问题。二是如何通过
求解使得对偶问题有优解的 $\alpha$ 转化为使原问题有最优解的 $w,b$

因为目标函数中
$$
y_i y_j(x_i\cdot x_j)
$$
仅与训练数据有关，而与优化变量 $\alpha$ 无关，
因此我们可以预先把这一部分计算出来。

数据集 $X$ 是 $N\times d$ 的矩阵，
标签 $y$ 是 $N\times 1$ 的列向量，考虑
$$G_x=XX^T$$
是数据的 Gram矩阵，而
$$G_y=yy^T$$
给出了标签之间的符号关系矩阵。
令 $$G = G_x \circ G_y$$是矩阵的 $\text{Hadamard}$ 乘积，于是我们
就得到了目标函数右侧的那一部分的矩阵。

于是目标函数变为了$$ f(\alpha) = \frac{1}{2} 
\sum_{i, j}^{N} \alpha_i \alpha_j G_{i,j}
-\sum_{i=1}^{N} \alpha_i$$
 这个本质上是二次型
$$f(\alpha) = \frac{1}{2} 
\alpha^T G\alpha -\sum_{i=1}^{N} \alpha_i$$其中 $G$ 是对称矩阵。

---

## 回到原问题

我们已经得到了 $\alpha$ 是对偶问题的解。现在我们要把它转化为原问题的解。

对于原问题，由 $\text{KKT}$ 条件可以得到 $ w = \sum_{i=1}^N \alpha_i y_i x_i$
这里 $\alpha$ 是通过计算得到的结果， $x$ 和 $y$ 是已知的样本数据，
从而可以计算得到 $w$.

而对 $b$ 的处理就要稍微麻烦一点。可以证明 $\alpha$ 不为零向量，否则
可以得到 $w=0$，从而其作为系数诱导
的分类函数 $f(x) = \text{sign}(wx+b)$ 将没有意义。因此必然存在 $\alpha_j >0 $，
我们对这一点进行分析。由于 $\alpha_j > 0$， 利用 $\text{KKT}$条件，
因为  $\alpha_j c_j(x)=0$，从而不等式约束 $c_j(x)$ 变为等式约束，也就是说 
$$c_j(x)=0$$

注意到 $$c_j(x) = 1-y_j(w\cdot x_j+b)=0$$，带入 $w$ 得到
$$ 1=y_j\left(\sum_{i=1}^N \alpha_i y_i (x_i\cdot x_j)+b\right)$$
并且由于 $y_j^2 = 1$，所以
$$\begin{aligned}
b &= y_j - \sum_{i=1}^{N}\alpha_iy_i(x_i \cdot x_j)\\
&=y_j\left(1- \sum_{i=1}^{N} \alpha_iG_{i,j} \right) 
\end{aligned}$$

或者我们直接从
$$c_j(x)=0$$推出
$$b=y_i\left( 1 - w\cdot x_j\right)$$也可以。

这样就得到了原问题的解。

---
## 一个看似可行的思路
现在关键的问题是，我们如何在约束 $$\sum_{i=1}^{N} \alpha_i 
y_i =0 ,\alpha_i \geqslant0$$ 下求出极值？
这里就要引入利用投影的梯度下降法。我们仍然是对 $f$ 求梯度，得到 $$\nabla f = G\alpha-1$$

由于函数梯度的方向指向其函数值增大的方向，所以我们要沿着其反方向做参数的更新。
设学习率 $\text{learning rate} = \eta$，
从而 $$\alpha^{new} = \alpha -\eta\nabla f$$

但是会导致我们不一定能够满足约束条件中的等式约束和不等式约束了。这怎么办呢？
这里有两种思路。
### 思路一
对于不等式约束，我们强制对每一个 $\alpha_i$取正值，令
$$\alpha^{pos} = \max\{\alpha, 0\}$$

对于等式约束，我们考虑向量在超平面上的投影向量。
$$
\alpha_{\parallel}=
\alpha^{pos}-
\frac{y^T\alpha^{pos}}{y^Ty}y
$$

### 思路二
我们先满足等式约束，我们考虑向量在超平面上的投影向量。
$$
\alpha_{\parallel}=
\alpha-
\frac{y^T\alpha}{y^Ty}y
$$
再考虑不等式约束，我们强制对每一个分量取正值，令
$$\alpha^{pos} = \max\{\alpha_{\parallel}, 0\}$$

---
上述过程几何意义其实非常直观。梯度下降会让参数沿着目标函数下降最快的方向移动，

但这个方向可能会离开约束平面，这里我们进行了两种投影方式，
第一种是先满足不等式约束再满足等式约束，
第二种是先满足等式约束再满足不等式约束。

---
$$
\boxed{
\begin{aligned}
G_x &= XX^T \\ 
G_y &= yy^T \\
G   &= G_x \circ G_y \\
f(\alpha)&=\frac12\alpha^T G\alpha-\mathbf1^T\alpha \\
\nabla f(\alpha) &=G\alpha-\mathbf1 \\
 \alpha  &= \text{project}(\alpha)                 \\
w &=\sum_{i=1}^{N}\alpha_i y_i x_i \\
b &=y_j\left(1- w \cdot x_j\right),\quad \alpha_j>0
\end{aligned}
}
$$

这个过程看上去可行，其实是有不少漏洞。
例如算法的收敛性证明，学习率的选择，包括两种不同投影顺序的区别，这些都要仔细考察。

因为我们想一次性处理所有的向量参数，但同时又要满足两个约束，这是不能做到的。
我们只能做到满足其中一个约束。这也是为什么我们需要引入新的迭代算法例如 
$\text{SMO}$等等。

不过我们还是可以分析一下两种算法哪一个更合理，以及学习率和收敛性的简单解释。
目前尚未找到能严格适用于本算法的收敛性定理，但数值实验中算法表现稳定，这一点可以结合不动点定理给出一些直观的解释。

---

## 关于两个投影方法的比较

两种不同的思路本质上是在对 $\text{SVM}$ 方法做一个回答。并且两次投影的顺序是不可以改的。
实际上，可以通过计算机进行验证，
在最后一步选择投影到非负象限，而不是选择投影到超平面，在相同的迭代次数之下，
算法的准确率和收敛性相对更优秀。

这个结果其实不难理解。在 SVM 对偶问题的两个约束条件中，不等式约束 
$$\alpha_i \geqslant 0 $$ 比等式约束 $$y^T\alpha = 0 $$
更为根本。一旦某个 $$\alpha_i$$ 变成负数，
拉格朗日对偶的下界性质就被破坏，后续的迭代也就失去了理论基础。因此必须强制保证 
$$\alpha_i \geqslant 0$$让非负截断成为最后一步。

这也恰好反映出 SVM 的核心思想。
SVM 关注的是那些对分类边界起关键作用的特殊样本——支撑向量。
支撑向量正是通过 $$ \alpha_i > 0 $$ 来筛选的。
如果最后一步破坏了这个条件，支撑向量就无法被正确地识别出来。


## 投影公式的简单证明
由于约束
$$\sum_{i=1}^{N}\alpha_i y_i=0$$可以写成
$$y^T\alpha=0$$
而这就是一个超平面，在 $2$ 维空间中体现为一条直线，在 $3$ 维空间中体现为一个平面。

对于向量 $\alpha^{pos}$，我们考虑其在超平面上的投影公式。这里简单证明一下。

考虑超平面
$$y^T\alpha=0$$
其中 $y$ 是该超平面的法向量。因此我们可以把 $\alpha^{pos}$ 分解为

$$\alpha^{pos}=\alpha_{\parallel}+\alpha_{\perp}$$

其中：

* $\alpha_{\parallel}$ 平行于超平面，也就是落在超平面内部
* $\alpha_{\perp}$ 垂直于超平面，也就是沿着法向量 $y$ 的方向

由于法向量方向唯一，因此存在某个实数 $\lambda$ 使得

$$\alpha_{\perp}=\lambda y$$
于是
$$\alpha^{pos}=\alpha_{\parallel}+\lambda y$$
因为 $\alpha_{\parallel}$ 在超平面上，所以满足
$$y^T\alpha_{\parallel}=0$$
对上式两边左乘 $y^T$ 得到
$$y^T\alpha^{pos}=y^T\alpha_{\parallel}+\lambda y^Ty$$
由$$y^T\alpha_{\parallel}=0$$可得
$$y^T\alpha^{pos}=\lambda y^Ty$$
因此$$\lambda=\frac{y^T\alpha^{pos}}{y^Ty}$$
从而垂直分量为$$\alpha_{\perp}=\frac{y^T\alpha^{pos}}{y^Ty}y$$
因此其在超平面上的投影为
$$
\boxed{
\alpha_{\parallel}=
\alpha^{pos}-
\frac{y^T\alpha^{pos}}{y^Ty}y
}
$$
这个也是我们更新参数的目标。于是对参数 $\alpha$，首先我们进行梯度下降法 $\alpha^{new}=\alpha - \eta \nabla f$，
接着剔除负数，令 $\alpha^{pos}= \max \{\alpha, 0\}$，最后将它投影在超平面上。令
$\alpha_{\parallel}=\alpha^{pos}-\dfrac{y^T\alpha^{pos}}{y^Ty}y$就得到最终结果。
