import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso

# 定义ReLU函数
def relu(x):
    return np.maximum(0, x)

# 目标函数：计算拟合误差
def objective_function(coeffs):
    # coeffs是一个多项式的系数
    # 通过多项式模型拟合ReLU函数
    def polynomial(x, coeffs):
        return sum(c * (x ** i) for i, c in enumerate(coeffs))
    
    # 在区间[-3, 3]内取一些点
    x_vals = np.linspace(-3, 3, 100)
    y_vals_true = relu(x_vals)  # 真实的ReLU函数值
    y_vals_pred = polynomial(x_vals, coeffs)  # 多项式拟合值
    
    # 计算误差：平方误差
    error = np.sum((y_vals_true - y_vals_pred) ** 2)
    return error

# 约束条件：系数范围
lb = [-10] * 4  # 假设多项式的最大次数为3，系数的下界
ub = [10] * 4   # 系数的上界

# 使用PSO算法进行优化
best_coeffs, _ = pso(objective_function, lb, ub, swarmsize=50, maxiter=100)

# 打印优化后的系数
print("Optimized Polynomial Coefficients:", best_coeffs)

# 打印多项式格式函数
def print_polynomial_format(coeffs):
    polynomial_str = ""
    for i, c in enumerate(coeffs):
        if i == 0:
            polynomial_str += f"{c}"
        elif i == 1:
            polynomial_str += f" + {c}x"
        else:
            polynomial_str += f" + {c}x^{i}"

    return polynomial_str

# 输出多项式格式
polynomial_function = print_polynomial_format(best_coeffs)
print(f"Fitted Polynomial: {polynomial_function}")

# 画图比较
x_vals = np.linspace(-3, 3, 100)
y_vals_true = relu(x_vals)
y_vals_pred = sum(c * (x_vals ** i) for i, c in enumerate(best_coeffs))

# 绘制结果图
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals_true, label='ReLU Function', color='blue')
plt.plot(x_vals, y_vals_pred, label='Fitted Polynomial', color='red', linestyle='dashed')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('ReLU Function and Fitted Polynomial')
plt.grid(True)

# 保存图像到指定路径
plt.savefig('/workspace/examples/vision_transformer/relu_particles1_approximation.png', dpi=300)
print("Image saved as 'relu_particles1_approximation.png'")

# 显示图像
plt.show()