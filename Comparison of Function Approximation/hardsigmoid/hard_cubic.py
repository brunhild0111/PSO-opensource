import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def hardsigmoid(x):
    return np.piecewise(x, [x <= -2.5, (-2.5 < x) & (x < 2.5), x >= 2.5],
                        [0, lambda x: (x + 2.5) / 5, 1])


x_interp_points = np.linspace(-3, 3, 10)
y_interp_points = hardsigmoid(x_interp_points)

cs = CubicSpline(x_interp_points, y_interp_points)

x_plot = np.linspace(-3, 3, 100)
y_plot = cs(x_plot)

plt.plot(x_plot, hardsigmoid(x_plot), label='hardsigmoid function')
plt.plot(x_plot, y_plot, label='Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approximation of hardsigmoid function using Cubic Spline Interpolation')
plt.legend()

# 指定保存图片的路径及文件名，这里需要确保路径存在，否则会保存失败
plt.savefig('/workspace/examples/vision_transformer/hardsigmoid_vs_cubic.png')

plt.show()
