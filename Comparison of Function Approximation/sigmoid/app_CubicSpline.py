import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import math


# 定义Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 生成数据点
x_data = np.linspace(-3, 3, 10)  # 选择10个数据点，可根据需要调整
y_data = sigmoid(x_data)

# 进行样条插值
cs = CubicSpline(x_data, y_data)

# 生成用于绘制近似曲线的更多数据点
x_plot = np.linspace(-3, 3, 1000)
y_plot = cs(x_plot)

# 打印近似多项式（实际上三次样条插值是分段多项式，这里以一种近似的方式展示每段的多项式系数）
for i in range(len(x_data) - 1):
    print(f"在区间 [{x_data[i]}, {x_data[i + 1]}] 上的近似多项式系数:")
    print(cs.c[:, i])

# 设置保存图片的路径
save_path = "/workspace/examples/vision_transformer/sigmoid_spline_approximation.png"

# 绘制Sigmoid函数和近似曲线并保存图片
plt.plot(x_plot, y_plot, label='Spline Interpolation Approximation')
plt.plot(x_data, y_data, 'ro', label='Data Points')
plt.plot(x_plot, sigmoid(x_plot), label='Sigmoid Function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spline Interpolation Approximation of Sigmoid Function')
plt.legend()
plt.savefig(save_path)
plt.show()
