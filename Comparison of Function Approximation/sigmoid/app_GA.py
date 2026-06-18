import numpy as np
import matplotlib.pyplot as plt


# 定义目标Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义多项式函数
def polynomial(x, coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))


# 遗传算法逼近Sigmoid函数
def genetic_algorithm_approximation(
    target_function, x_range, population_size=50, poly_degree=5, generations=100, mutation_rate=0.1
):
    # 初始化种群
    population = [np.random.uniform(-1, 1, poly_degree + 1) for _ in range(population_size)]

    # 定义适应度函数（均方误差）
    def fitness(individual):
        y_true = target_function(x_range)
        y_pred = polynomial(x_range, individual)
        return -np.mean((y_true - y_pred) ** 2)  # 越大越好

    # 选择操作
    def select(pop):
        scores = [fitness(ind) for ind in pop]

        # 将种群中的每个个体的系数拼接成一个长的一维数组
        flat_pop = np.concatenate([ind for ind in pop])

        # 将每个个体的适应度得分也按照个体系数的拼接方式进行拼接
        flat_scores = []
        for i in range(len(pop)):
            flat_scores.extend([scores[i]] * (len(pop[i])))
        flat_scores = np.array(flat_scores)

        selected = np.random.choice(
            flat_pop, size=len(pop), p=np.exp(flat_scores) / np.sum(np.exp(flat_scores))
        )

        # 将选择后的一维数组重新划分成原来个体的形状
        new_selected_population = []
        for i in range(0, len(selected), poly_degree + 1):
            new_selected_population.append(selected[i:i + poly_degree + 1])

        return new_selected_population

    # 交叉操作
    def crossover(parent1, parent2):
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    # 变异操作
    def mutate(individual):
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(len(individual))
            individual[idx] += np.random.uniform(-0.5, 0.5)
        return individual

    # 开始进化
    for gen in range(generations):
        # 选择和繁殖
        selected_population = select(population)
        next_population = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1 if i + 1 < len(selected_population) else 0]
            child1, child2 = crossover(parent1, parent2)
            next_population.extend([mutate(child1), mutate(child2)])
        population = next_population

        # 打印当前最佳适应度
        best_individual = max(population, key=fitness)
        print(f"Generation {gen + 1}: Best Fitness = {-fitness(best_individual):.6f}")

    # 返回最佳个体
    return best_individual


# 运行遗传算法
x_range = np.linspace(-3, 3, 100)
best_poly = genetic_algorithm_approximation(sigmoid, x_range)

# 按照类似1+x+3x^2的形式打印多项式
poly_str = ""
for i, coeff in enumerate(best_poly):
    if i == 0:
        a = 0
        poly_str += f"{coeff}"
    elif i == 1:
        poly_str += f"{coeff}x"
    else:
        poly_str += f"{coeff}x^{i}"
    if i < len(best_poly) - 1:
        poly_str += "+"
print("最佳多项式:", poly_str)

# 绘图对比
plt.figure(figsize=(8, 6))
plt.plot(x_range, sigmoid(x_range), label="Sigmoid Function", linewidth=2)
plt.plot(x_range, polynomial(x_range, best_poly), label="Polynomial Approximation", linestyle="--")
plt.legend()
plt.title("Sigmoid Function Approximation with Genetic Algorithm")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.savefig('/workspace/examples/vision_transformer/sigmoid_genetic_approximation.png', dpi=300)
plt.show()
