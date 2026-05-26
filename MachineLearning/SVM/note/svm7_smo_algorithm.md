# SMO算法

前面我们用投影的方法对SVM方法进行了计算。但是我们知道这样做是有缺陷的。
尽管在小范围的数据集上效果很好，但是速度比较慢。并且我们通过投影得到的结果不能同时都满足两个约束。
因此需要提出新的算法，这个算法叫做序列最小最优化算法(Sequential minimal optimization)。

对于原始问题
$$
\begin{aligned}
\min_{w,b,\xi}\quad
                 & \frac12\|w\|^2 + C\sum_{i=1}^N \xi_i \\ 
\text{s.t.}\quad & 1-\xi_i-y_i(w \cdot x_i+b)\le0 \\
                 & -\xi_i\le0
\end{aligned}
$$
可以写出拉格朗日函数
$$
\begin{aligned}
L(w, b, \xi, \alpha, \mu) = 
\frac12\|w\|^2+C\sum_{i=1}^{N}\xi_i &+ 
\sum_{i=1}^{N} \alpha_i\left(1-\xi_i-y_i(w\cdot x_i+b)\right) \\
&+\sum_{i=1}^{N}\mu_i(-\xi_i)
\end{aligned}
$$
我们有对偶问题
$$\begin{aligned}
\min_\alpha \quad & \frac12\sum_{i=1}^{N}\sum_{j=1}^{N}
\alpha_i\alpha_jy_iy_jK(x_i, x_j) - \sum_{i=1}^{N} \alpha_i \\
&= \frac12\alpha^T G \alpha - \sum_{i=1}^{N}\alpha_i\\
\text{s.t.} \quad & \sum_{i=1}^{N} \alpha_iy_i = 0\\
            \quad & 0 \leqslant \alpha_i \leqslant C
\end{aligned}$$
其中 $K(x,z)$ 是核函数，它代替了之前的内积。但仍满足线性性核非负性。

SMO的优势就是能够计算出解析解。我们通过每次只优化一个维度的变量实现这一目的。

这个算法是如何实现的呢？我们对原问题的参数 $\alpha$，如果初始状态下已经满足 $\alpha$ 
就是满足 KKT 条件的，那么这个 $\alpha$ 就已经是最终结果了。否则一定可以找到一个不满足 KKT 条件的点，
我们通过调整这个变量使该问题更接近最优解。
但因为一次只调整一个 $\alpha_i$ 必然会使等式约束失效，所以我们需要一次性调整两个参数。
此时我们可以用**可解析**的方法求出这个子问题。接下来就是这个具体的求解过程。

---
## 对问题的仔细分析
考虑
$$\begin{aligned}
\min_{\alpha_1, \alpha_2} \quad & W(\alpha_1, \alpha_2) =
\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2} K_{22}\alpha_2^2+y_1y_2K_{12}\alpha_1\alpha_2
-(\alpha_1+\alpha_2) \\
&\qquad\qquad\qquad+y_1 \alpha_1 \sum_{l \neq 1, 2} y_l\alpha_lK_{l1} 
+y_2\alpha_2 \sum_{l \neq 1, 2} y_l\alpha_lK_{l2} \\
\text{s.t.} \quad &  y_1\alpha_1+y_2\alpha_2 
            = -\sum_{l \neq 1, 2}^{N} \alpha_ly_l = \zeta \\
            \quad & 0 \leqslant \alpha_1 \leqslant C \\
            \quad & 0 \leqslant \alpha_2 \leqslant C
\end{aligned}$$
于是我们把原始问题转化为了只有两个变量的优化问题。
下面我们对 $\alpha_1$ 和 $\alpha_2$ 的值进行讨论。
因为 $y_1\alpha_1 + y_2\alpha_2 = \zeta$，从而
### 情形一 $y_1 \neq y_2$ 不是同一类
此时不妨设 $\alpha_1 - \alpha_2  = K$ 

故 $\alpha_2 = \alpha_1 - K$ 并且 $\alpha_1 \in [0, C]$, $\alpha_2 \in [-K, C-K]$

从而可得更新后的参数值的取值范围。令 $L = \max \{ 0, -K\} $，$H = \min
\{ C, C - K\}$
我们有
$$ L \leqslant \alpha_2^{\text{new}} \leqslant H $$
### 情形2 $y_1 = y_2$ 是同一类
此时我们有 $\alpha_1 + \alpha_2 = K$。
我们类似可以计算 $\alpha_2$ 的上界和下界。令 $L = \max \{ 0, K-C\}, H = \min \{ C, K\}$
得到
$$ L \leqslant \alpha_2^{\text{new}} \leqslant H$$

---

由于 $ \alpha_1y_1 = -\alpha_2y_2 + \zeta$，所以 $\alpha_1 = y_1(\zeta - \alpha_2y_2)$

从而可以代入目标函数进行计算。这是一个二次函数的问题，极值点是显然的。令


由于 $\alpha_1 y_1 + \alpha_2 y_2 = \zeta$，所以 $\alpha_1 = y_1(\zeta - \alpha_2 y_2)$。



则目标函数写为

$$
\begin{aligned}
W(\alpha_1, \alpha_2) &=
\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2} K_{22}\alpha_2^2+y_1 y_2 K_{12}\alpha_1\alpha_2
-(\alpha_1+\alpha_2) \\
&\quad+y_1 \alpha_1 \sum_{l \neq 1, 2} y_l\alpha_lK_{l1} 
+y_2\alpha_2 \sum_{l \neq 1, 2} y_l\alpha_lK_{l2} \\
\end{aligned}
$$

将 $\alpha_1 = y_1(\zeta - \alpha_2 y_2)$ 代入。注意到 $y_1^2 = 1$，有

$$
\alpha_1 = y_1\zeta - y_1 y_2 \alpha_2
$$


带入上述式子并求偏导，得到
$$
\begin{aligned}
\left(K_{11}+K_{22} - 2K_{12}\right) \alpha_2 + y_2\zeta(K_{12}-K_{11}) + y_1y_2-y_2^2 \\
+y_2\sum_{l\neq 1, 2} \alpha_ly_l(K_{l2}-K_{l1})=0
\end{aligned}
$$

从而

$$
\alpha_2^{\text{new,unc}} = \frac{y_1^2-y_1y_2+y_2\zeta(K_{11}-K_{12})
+y_2\sum_{l\neq 1, 2} \alpha_ly_l(K_{l1}-K_{l2})
}{K_{11}+K_{22}-2K_{12}}
$$

最后令
$$
\alpha_2^{\text{new}} = \begin{cases}
H  & \alpha_2^{\text{new,unc}} > H\\ 
L  & \alpha_2^{\text{new,unc}} < L\\
\alpha_2^{\text{new,unc}} & \text{其他} \\
\end{cases}
$$
与
$$
\alpha_1^{\text{new}} = y_1\zeta - y_1y_2\alpha_2^{\text{new}}
$$
参数 $w$ 和 $b$ 的更新方式与前一个相同

---
### 进一步整理

我们给上面整个过程做一个提炼。首先令 
$$\boxed{\eta = K_{11}+K_{22}-2K_{12}}$$
表示分母，注意到 $K$ 是核函数，由和函数的性质，不妨设 $K(x,z) = \Phi(x) \cdot \Phi(z)$，于是
$$
\eta = (\Phi(1))^2 +(\Phi(2))^2 - 2\Phi(1)\Phi(2) = (\Phi(1)-\Phi(2))^2
$$
是正数。
令
$$\boxed{g(x) = \sum_{i=1}^{N} \alpha_iy_iK(x_i, x) + b}$$
表示我们最终的决策函数。并且我们可以事先把每一个$g(x_j)$ 进行计算。后面我们会看到这样处理的好处。令
$$\boxed{E_i = g(x_i) - y_i}$$
表示预测结果和实际结果的差。如果预测正确，那么
 $g(x_i)$
的符号应该和
 $y_i$ 
 相同。如果预测错误，符号就不同。
通过函数 $g(x)$ 和误差 $E_i$ 可以帮助我们简化上述表达式。首先我们有
$$\begin{aligned}
\sum_{l\neq 1, 2} \alpha_ly_l(K_{l1}-K_{l2})  &= \left( g(x_1) - y_1\alpha_1K_{11}-y_2\alpha_2K_{12}-b\right)
-\left(g(x_2) - y_1\alpha_1K_{21}-y_2\alpha_2K_{22}-b \right) \\
& = (g(x_1)-g(x_2))-y_1\alpha_1(K_{11}-K_{21})-y_2\alpha_2(K_{12}-K_{22})
\end{aligned}
$$

又因为在$$
\alpha_2^{\text{new,unc}} = \frac{y_2^2-y_1y_2+y_2\zeta(K_{11}-K_{12})
+y_2\sum_{l\neq 1, 2} \alpha_ly_l(K_{l1}-K_{l2})
}{K_{11}+K_{22}-2K_{12}}$$中，利用 $\zeta = y_1\alpha_1^{\text{old}} + y_2\alpha_2^{\text{old}}$
可以得到
$$\begin{aligned}
\alpha_2^{\text{new,unc}} ({K_{11}+K_{22}-2K_{12}}) &= y_2^2-y_1y_2+y_2(y_1\alpha_1^{\text{old}} + y_2\alpha_2^{\text{old}})(K_{11}-K_{12})\\ 
& \quad +y_2((g(x_1)-g(x_2))-y_1\alpha_1^{\text{old}}(K_{11}-K_{21})-y_2\alpha_2^{\text{old}}(K_{12}-K_{22}))\\
\\
&=y_2((g(x_1)-y_1)-(g(x_2)-y_2))+ \alpha_2^{\text{old}}(K_{11}+K_{22}-2K_{12})\\
\end{aligned}$$
从而
$$\boxed{
\alpha_2^{\text{new,unc}} = \alpha_2^{\text{old}} +\frac{y_2(E_1-E_2)}{\eta}}
$$
进而得到 $\alpha_2^{\text{new}}$。于是
$$\begin{aligned}
\alpha_1^{\text{new}}& = y_1\zeta - y_1 y_2 \alpha_2 \\
&= y_1(y_1\alpha_1^{\text{old}} + y_2\alpha_2^{\text{old}}) -y_1y_2\alpha_2^{\text{new}}\\
&= \alpha_1^{\text{old}} +y_1y_2 (\alpha_2^{\text{old}}-\alpha_2^{\text{new}} )
\end{aligned}
$$
于是
$$\boxed{
\alpha_2^{\text{new}} = \begin{cases}
H  & \alpha_2^{\text{new,unc}} > H\\ 
L  & \alpha_2^{\text{new,unc}} < L\\
\alpha_2^{\text{new,unc}} & \text{其他} \\
\end{cases}}
$$
$$\boxed {\alpha_1^{\text{new}} = \alpha_1^{\text{old}} + y_1y_2(\alpha_2^{\text{old}} - \alpha_2^{\text{new}}) }$$


---


## KKT条件的分析以及变量的选择流程
上面只是分析了 $\alpha$ 的迭代更新过程，我们现在还要讨论 $\alpha$ 是如何选择的，同时我们还要分析如何

由于我们的KKT条件中，我们有驻点条件
$$
\begin{aligned}
\frac{\partial L}{\partial w}  &= w - \sum_{i=1}^{N}\alpha_i y_i x_i = 0 \\
\frac{\partial L}{\partial b}  &= -\sum_{i=1}^{N}\alpha_i y_i = 0 \\
\frac{\partial L}{\partial \xi_i} &= C - \alpha_i - \mu_i = 0
\end{aligned}
$$
得到
$$
\begin{aligned}
w = \sum_{i=1}^{N}\alpha_i y_i x_i \\
\sum_{i=1}^{N}\alpha_i y_i = 0 \\
\mu_i = C - \alpha_i
\end{aligned}
$$
可行性包括原始可行性和对偶可行性，要求
$$\begin{aligned}
 1-\xi_i-y_ig(x_i)& \leqslant 0 \\
 -\xi_i& \leqslant 0 \\
\alpha_i &\geqslant 0 \\
\mu_i &\geqslant 0 \\
\end{aligned}
$$
结合互补松弛条件
$$
\begin{aligned}
\alpha_i(1-\xi_i-y_ig(x_i)) &= 0 \\
\mu_i\xi_i &= 0
\end{aligned}
$$
其中 $g(x) = wx+b = \sum_{i=1}^{N} \alpha_iy_iK(x_i, x)+b $。
现在我们要选择最合适的 $\alpha$。首先考虑 KKT 条件是否满足。

我们分若干个情况分析以下

### Step 1 找 $\alpha_1$
首先我们要明确一点，我们要选择的 $\alpha$ 是最需要被优化的那个点。比如说一个因素它的偏差很大，那我肯定要首先去优化这个因素。
这里我就是要找“影响程度最大”的那个 $\alpha_i$。类比支撑向量的思路，这里我们也要寻找合适的 $\alpha_i$
有支撑向量的性质，也就是要找哪些不符合 KKT 条件的。

* 如果 $\alpha_i = 0$，从而利用驻点条件 $ \mu_i = C - \alpha_i$ 得到  $\mu_i = C$，因此从互补松弛条件 $ \mu_i\xi_i = 0$ 中，能够推出 $\xi_i = 0$。再利用原始可行性，得到 $y_ig(x_i) \geqslant 1$


* 如果 $\alpha_i = C$， 首先通过驻点条件可以得到 $\mu_i = 0$，从而结合互补松弛条件可以得到 $1-\xi_i-y_ig(x_i) = 0$。结合原始可行性 $\xi_i \geqslant 0$ 得到 $y_ig(x_i) \leqslant 1$

* 如果 $0 <\alpha_i<C$，结合驻点条件从而我们有 $\mu_i \neq 0$。因此利用互补松弛条件有 $\xi_i = 0$  所以 $y_ig(x_i) =1$

所以为了让不等式约束真的发挥作用，我们只能让 $\alpha$ 介于两者之间，也就是 $0 < \alpha < C$。或者让 KKT条件不成立，也就是 $\alpha=0$ 而 $y_ig(x_i) <1$ 与 $\alpha = C$ 而 $y_ig(x_i) >1$ 两种情况。

所以第一步，我们就能够先找到满足条件的 $\alpha_i$

### Step 2 找 $\alpha_2$
上面我们已经通过计算得到了
$$\alpha_2^{\text{new, unc}} = \alpha_2^{\text{old}} + \frac{y_2(E_1-E_2)}{\eta}$$
是和 $$|E_1-E_2|$$ 有关的。因此我们确定了 $\alpha_1$ 也就确定了 $E_1$，接下来尽可能要找使得 $$|E_1-E_2|$$ 尽可能多的 $\alpha_2$。

### Step 3 计算 $b$ 的值

根据 KKT条件 可以得到，如果 $0 < \alpha_1^{\text{new}} < C$，根据 **Step 1** 可以得到 $ y_1g(x_1) = 1$。因此展开得到
$$\sum_{i=1}^{N} \alpha_iy_iK_{i1} + b = y_1$$
也就是 $$b = y_1-\sum_{i=1}^{N} \alpha_iy_iK_{i1}$$
如果 $\alpha_1^{\text{new}}$ 不满足，而 $\alpha_2^{\text{new}}$ 满足 $0<\alpha_2^{\text{new}}<C$，我们有
$$b = y_2-\sum_{i=1}^{N} \alpha_iy_iK_{i2}$$

在实际程序中，我们可以通过之前已经计算过的 $E_1$、$E_2$ 以及 $g(x_1)$、$g(x_2)$ 进行处理。以 $$\sum_{i=1}^{N} \alpha_iy_iK_{i1} + b = y_1$$作为例子，由于
$$
\begin{aligned}
E_1 &= g(x_1) - y_1 \\
&= \left( \sum_{i=1}^{N} \alpha_iy_iK_{i1} +b\right) -y_1\\
&= \sum_{i=3}^{N} \alpha_iy_iK_{i1} + \alpha_1^{\text{old}}y_1K_{11}
+\alpha_2^{\text{old}}y_2K_{21} + b^{\text{old}} -y_1
\end{aligned}$$
得到
$$
\sum_{i=3}^{N} \alpha_iy_iK_{i1} =y_1 + E_1 - \alpha_1^{\text{old}}y_1K_{11}
- \alpha_2^{\text{old}}y_2K_{21}- b^{\text{old}} 
$$
再结合最开始计算的新的 $b$ 的式子，我们有
$$\begin{aligned} 
b_1^{\text{new}} &= y_1 -\sum_{i=1}^{N} \alpha_iy_kK_{i1}\\
&=y_1- \sum_{i=3}^{N} \alpha_iy_iK_{i1} - \alpha_1^{\text{old}}y_1K_{11}
-\alpha_2^{\text{old}}y_2K_{21} \\
&=y_1- \sum_{i=3}^{N} \alpha_iy_iK_{i1} - \alpha_1^{\text{new}}y_1K_{11}
-\alpha_2^{\text{new}}y_2K_{21} \\
&= -E_1 -y_1K_{11}(\alpha_1^{\text{new}}-\alpha_1^{\text{old}} ) -y_2K_{21}(\alpha_2^{\text{new}}-\alpha_2^{\text{old}}) + b^{\text{old}}
\end{aligned}$$
这样就只需要再我们已知的一些数据上进行更新就可以了。
同理可得，如果是第二种情况，我们有
$$b_2^{\text{new}}=-E_2 -y_1K_{12}(\alpha_1^{\text{new}}-\alpha_1^{\text{old}} ) -y_2K_{22}(\alpha_2^{\text{new}}-\alpha_2^{\text{old}}) + b^{\text{old}}
$$
并且如果两种情况同时成立，也就是
$$
0 < \alpha_1 , \alpha_2 < C
$$
两个算出来的 $b$ 应该是一致的，即 $b_1^{\text{new}} = b_2^{\text{new}}$。
如果不成立，那么我们取中点即可，因为此时只有不等式约束而没有等式约束。因此可以任意选取。




---