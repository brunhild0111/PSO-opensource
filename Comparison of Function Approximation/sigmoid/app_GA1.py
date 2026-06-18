import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# 定义目标Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义多项式函数
def polynomial(x, coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

# 初始化DEAP框架
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))  # 最小化均方误差
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
poly_degree = 1  # 多项式阶数
toolbox.register("attr_float", np.random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, poly_degree + 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 评估函数
def evaluate(individual, x_range):
    y_true = sigmoid(x_range)
    y_pred = polynomial(x_range, individual)
    return np.mean((y_true - y_pred) ** 2),  # 注意这里要返回一个元组

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate, x_range=np.linspace(-3, 3, 100))

# 遗传算法运行
def run_ga():
    population = toolbox.population(n=500)
    generations = 100

    # 使用 DEAP 的简单遗传算法流程
    result_pop, log = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=generations,
        stats=None,
        halloffame=None,
        verbose=True,
    )

    # 返回最佳个体
    best_individual = tools.selBest(result_pop, k=1)[0]
    return best_individual

best_poly = run_ga()

# 按照类似1+x+3x^2的形式打印多项式
poly_str = ""
for i, coeff in enumerate(best_poly):
    if i == 0:
        poly_str += f"{coeff:.3f}"
    elif i == 1:
        poly_str += f" + {coeff:.3f}x"
    else:
        poly_str += f" + {coeff:.3f}x^{i}"
print("最佳多项式:", poly_str)

# 绘图对比
x_range = np.linspace(-3, 3, 100)
plt.figure(figsize=(8, 6))
plt.plot(x_range, sigmoid(x_range), label="Sigmoid Function", linewidth=2)
plt.plot(x_range, polynomial(x_range, best_poly), label="Polynomial Approximation", linestyle="--")
plt.legend()
plt.title("Sigmoid Function Approximation with Genetic Algorithm")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

# 保存图片并打印多项式
save_path = '/workspace/examples/vision_transformer/sigmoid_genetic_approximation.png'
plt.savefig(save_path, dpi=300)
print(f"图片已保存到: {save_path}")
plt.show()
