import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 线性近似函数
def linear_approx(x, k):
    return k * x + 0.5

# 拟合误差函数
def fit_error(k):
    x = np.linspace(-3, 3, 1000)
    sigmoid_vals = sigmoid(x)
    linear_vals = linear_approx(x, k)
    return np.mean((sigmoid_vals - linear_vals) ** 2)

# 优化斜率 k
result = minimize(fit_error, x0=0.2, bounds=[(0, None)])
optimal_k = result.x[0]

# 打印优化结果（从高次到低次）
print("="*50)
print(f"Optimal k (slope): {optimal_k:.4f}")
print("="*50)

# 可视化结果
x = np.linspace(-3, 3, 1000)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, linear_approx(x, optimal_k), label=f'Linear Approx (k={optimal_k:.4f})')
plt.legend()
plt.title("Linear Approximation of Sigmoid")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

# 保存图片
plt.savefig('/workspace/examples/vision_transformer/hardsigmoid_linear.png')
plt.show()
