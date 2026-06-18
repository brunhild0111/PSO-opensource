import numpy as np
import matplotlib.pyplot as plt


# 定义修改后的hardsigmoid函数
def hardsigmoid(x):
    result = np.zeros_like(x)
    mask_1 = x <= -2.5
    mask_2 = (x > -2.5) & (x < 2.5)
    mask_3 = x >= 2.5

    result[mask_1] = 0
    result[mask_2] = 0.2 * x[mask_2] + 0.5
    result[mask_3] = 1

    return result


def monte_carlo_approximation(func, a, b, num_points):
    x_samples = np.random.uniform(a, b, num_points)
    y_samples = func(x_samples)

    poly_degree = 3  # You can adjust the degree of the polynomial as needed
    poly_fit = np.polyfit(x_samples, y_samples, poly_degree)
    poly_func = np.poly1d(poly_fit)

    return poly_func


# Set the interval and the number of sampling points
a = -3
b = 3
num_points = 10000

# Get the approximated polynomial function
approx_poly_func = monte_carlo_approximation(hardsigmoid, a, b, num_points)

# 获取多项式系数
coefs = approx_poly_func.coefficients

# 按照指定格式打印近似多项式
print("Approximated polynomial:")
for i, coef in enumerate(coefs):
    if i == 0:
        print(f"{coef}", end="")
    elif i == 1:
        print(f" + {coef}x", end="")
    else:
        print(f" + {coef}T_{i}(x)", end="")
print()

# Generate x values for plotting the graph
x_plot = np.linspace(a, b, 1000)

# Calculate the values of the original hardsigmoid function
y_original = hardsigmoid(x_plot)

# Calculate the values of the approximated polynomial function
y_approx = approx_poly_func(x_plot)

# Plot the graph
plt.plot(x_plot, y_original, label='hardsigmoid function')
plt.plot(x_plot, y_approx, label='Approximated polynomial function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('hardsigmoid function and its Monte Carlo approximation')
plt.legend()

# Save the image
plt.savefig('/workspace/examples/vision_transformer/hardsigmoid_monte_approximation.png')

# Show the image
plt.show()
