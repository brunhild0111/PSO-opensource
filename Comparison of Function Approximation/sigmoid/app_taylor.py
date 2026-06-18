import numpy as np
import matplotlib.pyplot as plt


# Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 泰勒级数逼近sigmoid函数（取到三阶导数）
def taylor_approximation(x):
    return 0.5 + 0.25 * x - 0.021 * x ** 3


# 生成[-3, 3]区间的x值
x_values = np.linspace(-3, 3, 100)

# 计算真实sigmoid函数值
sigmoid_values = sigmoid(x_values)

# 计算泰勒逼近值
taylor_values = taylor_approximation(x_values)

# 绘制图像
plt.plot(x_values, sigmoid_values, label='Sigmoid Function')
plt.plot(x_values, taylor_values, label='Taylor Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sigmoid Function and Taylor Approximation')
plt.legend()
# 保存图片
plt.savefig('/workspace/examples/vision_transformer/Taylor_approximation.png', dpi=300)
plt.show()
