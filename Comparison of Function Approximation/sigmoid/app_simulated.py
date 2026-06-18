import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Polynomial function
def polynomial(x, coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

# Error function
def error(coeffs, x_range):
    func = lambda x: (sigmoid(x) - polynomial(x, coeffs))**2
    return quad(func, x_range[0], x_range[1])[0]

# Simulated annealing algorithm
def simulated_annealing(degree, x_range, max_iter=1000, initial_temp=1.0, cooling_rate=0.99):
    np.random.seed(0)
    # Initialize coefficients randomly
    coeffs = np.random.uniform(-1, 1, degree + 1)
    best_coeffs = coeffs.copy()
    best_error = error(best_coeffs, x_range)
    temp = initial_temp

    for _ in range(max_iter):
        # Randomly perturb one coefficient
        new_coeffs = coeffs.copy()
        idx = np.random.randint(0, degree + 1)
        new_coeffs[idx] += np.random.uniform(-0.1, 0.1)

        # Calculate new error
        new_error = error(new_coeffs, x_range)
        _error = new_error - best_error

        # Decide whether to accept the new solution
        if _error < 0 or np.random.rand() < np.exp(-_error / temp):
            coeffs = new_coeffs
            if new_error < best_error:
                best_error = new_error
                best_coeffs = new_coeffs

        # Cool down
        temp *= cooling_rate

    return best_coeffs

# Main process
x_range = [-3, 3]
degree = 3
coeffs = simulated_annealing(degree, x_range)

# Generate plot
x = np.linspace(x_range[0], x_range[1], 500)
y_sigmoid = sigmoid(x)
y_poly = polynomial(x, coeffs)

plt.figure(figsize=(8, 6))
plt.plot(x, y_sigmoid, label="Sigmoid Function", linewidth=2)
plt.plot(x, y_poly, label=f"Polynomial Approximation (degree={degree})", linestyle="--")
plt.title("Polynomial Approximation of Sigmoid Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
# 修改保存图片路径和dpi
plt.savefig('/workspace/examples/vision_transformer/sigmoid_simulated_approximation.png', dpi=300)
plt.show()

# 按照类似1+x+3x^2的形式打印多项式
polynomial_str = ""
for i, c in enumerate(coeffs):
    if i == 0:
        polynomial_str += f"{c}"
    elif i == 1:
        polynomial_str += f"{c}x"
    else:
        polynomial_str += f"{c}x^{i}"
    if i < len(coeffs) - 1:
        polynomial_str += "+"

print("Approximated Polynomial Coefficients:")
print(polynomial_str)
