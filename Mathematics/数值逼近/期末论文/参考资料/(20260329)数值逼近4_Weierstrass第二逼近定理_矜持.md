# 数值逼近（4） Weierstrass第二逼近定理

 **Author:** [矜持]

 **Link:** [https://zhuanlan.zhihu.com/p/2021566370396079053]

[数值逼近（1） Weierstrass第一逼近定理](https://zhuanlan.zhihu.com/p/2020938535826405077)[数值逼近（2） Baskakov线性正算子序列](https://zhuanlan.zhihu.com/p/2020956774539239731)[数值逼近（3）Mirakyan算子序列](https://zhuanlan.zhihu.com/p/2021270561574134503)
> 前面我们用 Bernstein 多项式证明了 Weierstrass 第一逼近定理，这里我们引入 Vallee-Poussin 算子证明 Weierstrass 第二逼近定理。同样，这里我们给出构造性证明。

---

定义 V-P 算子 $V_n(f;x)$$V_n(f;x) = \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{-\pi}^{\pi} f(t)\cos ^{2n} \dfrac{t-x}{2} dt$

其中 $f$ 是周期为 $2\pi$ 的连续函数。首先我们说明 V-P 算子可以转化为三角多项式。因为

$\cos \frac{t-x}{2} = \frac{e^{i\tfrac{t-x}{2}} + e^{-i\tfrac{t-x}{2}}}{2}$ 故 由二项式定理$$\cos^{2n} \frac{t-x}{2} = \frac{1}{2^{2n}} \sum_{k=0}^{2n} \binom{2n}{k} \left( e^{i\tfrac{t-x}{2}} \right)^{2n-k} \left( e^{-i\tfrac{t-x}{2}} \right)^k$$ 带入积分式子得到：$$V_n(f; x) = \frac{C_n}{2^{2n}} \sum_{k=0}^{2n} \binom{2n}{k} \int_{-\pi}^{\pi} f(t) e^{i(n-k)(t-x)} dt$$ 其中 $C_n$ 为系数。对于有限量，求和和积分可以交换次序。因此

$$V_n(f; x) = \sum_{k=0}^{2n} \left[ \frac{C_n \binom{2n}{k}}{2^{2n}} \int_{-\pi}^{\pi} f(t) e^{i(n-k)t} dt \right] e^{-i(n-k)x}$$ 令 $j=n-k $$$V_n(f; x) = \sum_{j=-n}^{n} \left[ \frac{C_n \binom{2n}{n-j}}{2^{2n}} \int_{-\pi}^{\pi} f(t) e^{ijt} dt \right] e^{-ijx}$$中间是与 $t$ 无关的积分结果，因此我们得到 $$V_n(f; x) = \sum_{j=-n}^{n} c_j e^{-ijx}$$

其中 $$c_j = \dfrac{C_n \binom{2n}{n-j}}{2^{2n}} \int_{-\pi}^{\pi} f(t) e^{ijt} dt$$ 注意到 $  c_{-j} = \dfrac{C_n \binom{2n}{n+j}}{2^{2n}} \int_{-\pi}^{\pi} f(t) e^{-ijt} dtj=\overline{c_j}$是共轭对称的。从而

$$c_j e^{-ijx} + c_{-j} e^{ijx} = c_j e^{-ijx} + \overline{c_j} e^{ijx} = 2 \text{Re}(c_j e^{-ijx})$$ 因此这里不存在虚数，全部可以用三角多项式来表示。因此上面定义的 V-P 算子就是所构造的三角多项式。下面开始证明。

---

由 Wails 公式，注意到 $I_n = \int_{-\pi}^{\pi} \cos ^{2n} \dfrac{t-x}{2} dt\\= \int_{-\pi}^{\pi} \cos ^{2n} \dfrac{t}{2} dt \\=2\int_{0}^{\pi} \cos ^{2n} \dfrac{t}{2} dt\\ =4\int_{0}^{\frac{\pi}{2}} \cos ^{2n} u  du\\ =\dfrac{2\pi\cdot(2n-1)!!}{(2n)!!} $ 刚好为 $V_n(f;x)$ 前面的系数的倒数，从而我们有

$f(x) = \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{-\pi}^{\pi} f(x)\cos ^{2n} \dfrac{t-x}{2} dt$ 做差得 $\left|V_n(f;x)-f(x) \right| = \left| \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{-\pi}^{\pi} \left(f(t)-f(x)\right)\cos ^{2n} \dfrac{t-x}{2} dt\right| \\ \leq   \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{-\pi}^{\pi} \left|f(t)-f(x)\right|\cos ^{2n} \dfrac{t-x}{2} dt$ 由 $\mathbb{R}$ 上周期为 $2\pi$ 的连续函数的一致连续性，我们得到

$\forall \epsilon >0,\exists \delta >0,\forall |x - y|<\epsilon, x,y\in \mathbb{R}, \left| f(x) - f(y) \right| < \epsilon$ 于是上式可化为

$  \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{I_1 \cup I_2} \left|f(t)-f(x)\right|\cos ^{2n} \dfrac{t-x}{2} dt$ 其中 $I_1 = [-\pi,\pi] \cap [x-\delta, x+\delta] $ ， $I_2 = [-\pi, \pi] - I_1$

令 $I = I_1 \cup I_2 = [-\pi, \pi]$ 。我们对区间 $I_1$ 讨论有

$  \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{I_1 } \left|f(t)-f(x)\right|\cos ^{2n} \dfrac{t-x}{2} dt  \\ \leq     \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{I_1 } \epsilon\cos ^{2n} \dfrac{t-x}{2} dt  \\  \leq \epsilon   \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{I}\cos ^{2n} \dfrac{t-x}{2} dt   =\epsilon$ 对区间 $I_2$ ，我们利用 $|x-t|\geq \delta$ ，得到 $\left| \cos \dfrac{t-x}{2} \right| \leq \left| \cos \dfrac{\delta}{2} \right|$ 。令 $M = \sup _{x\in\mathbb{R}} |f(x)|$ ，我们得到

$  \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{I_2 } \left|f(t)-f(x)\right|\cos ^{2n} \dfrac{t-x}{2} dt   \\ \leq   \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{I } \left|f(t)-f(x)\right|\cos^{2n} \dfrac{\delta}{2} dt   \\  \leq   \dfrac{(2n)!!}{2\pi\cdot(2n-1)!!}\int_{I } 2M\cos^{2n} \dfrac{\delta}{2} dt    \\  =\dfrac{2M\cdot (2n)!! \cdot \cos ^{2n} \dfrac{\delta}{2} \cdot 2\pi}{2\pi\cdot(2n-1)!!}  \\ =\dfrac{2M\cdot(2n)!!\cdot \cos^{2n} \dfrac{\delta}{2}}{(2n-1)!!}$ 这里利用归纳法可以证明 $\dfrac{(2n)!!}{(2n-1)!!} \leq 2n$ ,从而上述积分式 $\leq4Mn \cos^{2n} \dfrac{\delta}{2}$ 。

由于 $\cos^{2n} \dfrac{\delta}{2} <1$ ，存在 $N$ 使得当 $n>N$ 时，上式 $<\epsilon$

因此 $\left|V_n(f;x)-f(x) \right|<2\epsilon$ 对任意的 $x\in \mathbb{R}$ 成立。从而我们完成了 Weierstrass 第二逼近定理的证明。

---

Note：我们还可以用Weierstrass第一逼近定理诱导Weierstrass第二逼近定理的证明。方法有很多。