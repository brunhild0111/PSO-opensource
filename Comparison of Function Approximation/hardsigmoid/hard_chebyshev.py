import numpy as np
import matplotlib.pyplot as plt


# 新的hardsigmoid函数定义
def hardsigmoid(x):
    result = np.zeros_like(x)
    mask1 = x <= -2.5
    mask2 = (-2.5 < x) & (x < 2.5)
    mask3 = x >= 2.5

    result[mask1] = 0
    result[mask2] = 0.2 * x[mask2] + 0.5
    result[mask3] = 1

    return result


# 切比雪夫多项式定义
def chebyshev_polynomial(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return 2 * x * chebyshev_polynomial(n - 1, x) - chebyshev_polynomial(n - 2, x)


# 切比雪夫逼近步骤
def chebyshev_approximation(N, a, b):
    # 将区间 [-3, 3] 映射到 [-1, 1] 的系数
    alpha = (b + a) / 2
    beta = (b - a) / 2

    # 在 [-1, 1] 上选取离散点，改为取N + 1个点
    x_cheb = np.cos((np.pi * 2 * np.arange(N + 1)) / (2 * (N + 1)))

    # 映射回原始区间 [-3, 3]
    x = alpha + 2 * beta * x_cheb

    # 计算hardsigmoid函数在这些点的值
    y_hardsigmoid = hardsigmoid(x)

    # 计算切比雪夫系数
    cheb_coeffs = np.zeros(N + 1)
    for k in range(N + 1):
        cheb_coeffs[k] = (2 / (N + 1)) * np.sum(y_hardsigmoid * chebyshev_polynomial(k, x_cheb))

    return cheb_coeffs


def chebyshev_approximate_function(cheb_coeffs, a, b):
    alpha = (b + a) / 2
    beta = (b - a) / 2

    def approximate(x):
        t = (2 * x - (b + a)) / (b - a)
        approx_value = cheb_coeffs[0] / 2
        for k in range(1, len(cheb_coeffs)):
            approx_value += cheb_coeffs[k] * chebyshev_polynomial(k, t)
        return approx_value

    return approximate


# 示例用法
# 选择逼近的阶数
N = 3
print("N:", N)

a = -3
b = 3
# 计算切比雪夫系数
cheb_coeffs = chebyshev_approximation(N, a, b)

# 构建逼近函数
approximate_hardsigmoid = chebyshev_approximate_function(cheb_coeffs, a, b)

# 打印拟合hardsigmoid的多项式
polynomial_str = ""
for i, coeff in enumerate(cheb_coeffs):
    if i == 0:
        polynomial_str += f"{coeff / 2}"
    elif i == 1:
        polynomial_str += f" + {coeff}x"
    else:
        polynomial_str += f" + {coeff}T_{i}(x)"

print("拟合hardsigmoid的多项式表达式：", polynomial_str)

# 在一些点上测试逼近效果
x_test = np.linspace(-3, 3, 100)
y_hardsigmoid_test = hardsigmoid(x_test)
y_approx_test = np.array([approximate_hardsigmoid(x) for x in x_test])

# 画图
plt.plot(x_test, y_hardsigmoid_test, label='hardsigmoid', linewidth=2)
plt.plot(x_test, y_approx_test, label=f'Chebyshev Approximation (N={N})', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Chebyshev Approximation of hardsigmoid Function')
plt.legend()

# 设置图片保存路径及文件名
save_path = '/workspace/examples/vision_transformer/hardsigmoid_chebyshev_approximation.png'
plt.savefig(save_path, dpi=300)

# 显示图片（可选，如果不需要在程序运行时显示图片可注释掉这行）
plt.show()
