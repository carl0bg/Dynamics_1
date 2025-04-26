from math import sin, pi
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tabulate import tabulate
import time as time_module


def create_grid(length, total_time, n, m, u0, mu):
    grid = []
    h = length / n
    t = total_time / m
    for j in range(m):
        grid.append([0] * n)
    for i in range(n):
        grid[0][i] = u0(i * h)
    for j in range(m):
        grid[j][0] = mu(j * t)
    return grid, h, t


def u0_0(x):
    return float(x >= 1 and x <= 2)  # прямоугольный импульс


def u0_1(x):
    if x < 1 or x > 2:
        return 0
    elif x >= 1 and x < 1.5:
        return 2 * (x - 1)
    else:
        return 1 - 2 * (x - 1.5)  # треугольный импульс


def u0_2(x):
    if x < 1 or x > 2:
        return 0
    else:
        return 0.5 * (1 + sin(2 * pi * (x - 1) - pi / 2))  # синусоидальный импульс


def u0_3(x):
    if x < 0 or x > 1:
        return 0
    else:
        return 1 - x**2  # функция для половины параболы


def mu(t):
    return 0


def solve1(grid, a, h, tau):
    for t in range(1, len(grid)):
        for x in range(1, len(grid[t])):
            grid[t][x] = grid[t - 1][x] - a * tau * ((grid[t - 1][x] - grid[t - 1][x - 1]) / h)
    return grid


def solve2(grid, a, h, tau):
    for t in range(1, len(grid)):
        for x in range(1, len(grid[t]) - 1):
            grid[t][x] = 0.5 * (grid[t - 1][x + 1] + grid[t - 1][x - 1]) \
                         - a * tau / (2 * h) * (grid[t - 1][x + 1] - grid[t - 1][x - 1])
    return grid


def solve_single(t, grid, a, h, tau):
    prev = grid[t - 1]
    curr = grid[t]
    mu1 = curr[0]
    mu2 = 0
    kappa1 = 0
    kappa2 = 0
    ai = [kappa1]
    bi = [mu1]
    C = 1
    A = -a * tau / (4 * h)  # Для Кранка-Николсон берём половину центральной разности
    B = a * tau / (4 * h)
    for x in range(1, len(curr) - 1):
        # Правая часть для Кранка-Николсон: добавляем (u_{i+1}^n - u_{i-1}^n)/(2h)
        phi = prev[x] - (a * tau / (4 * h)) * (prev[x + 1] - prev[x - 1])
        denom = C + A * ai[x - 1]
        alpha = -B / denom
        beta = (phi - A * bi[x - 1]) / denom
        ai.append(alpha)
        bi.append(beta)
    curr[-1] = (kappa2 * bi[-1] + mu2) / (1 - kappa2 * ai[-1])
    for x in range(len(curr) - 2, -1, -1):
        curr[x] = ai[x] * curr[x + 1] + bi[x]


def solve3(grid, a, h, tau):
    for t in range(1, len(grid)):
        solve_single(t, grid, a, h, tau)
    return grid


def generate_precise(u0, a, i, h, n, tau):
    grid = []
    for x in range(n):
        grid.append(u0(x * h - i * tau * a))
    return grid


def print_comparison_table(step, x_vals, method1, method2, precise, h_x, h_t):
    step_size = 5
    indices = range(0, len(x_vals), step_size)
    
    table_data = []
    headers = ["x (шаг h_x={:.4f})".format(h_x), "Method 1", "Method 2", "Precise", "delta 1", "delta 2"]
    
    for i in indices:
        x = x_vals[i]
        m1 = method1[i]
        m2 = method2[i]
        pr = precise[i]
        error1 = abs(m1 - pr)
        error2 = abs(m2 - pr)
        table_data.append([
            f"{x:.4f}",
            f"{m1:.4f}", 
            f"{m2:.4f}", 
            f"{pr:.4f}", 
            f"{error1:.4f}", 
            f"{error2:.4f}"
        ])
    
    print(f"\nСравнение на шаге {step} (t = {step*h_t:.2f}, h_t = {h_t:.4f}):")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))


a = 2
w = 15
total_time = 10  # Переименовываем переменную, чтобы избежать конфликта с модулем time
n = w * 120  # число пространственных узлов
m = total_time * 250  # число временных шагов
h_x = w / n
h_t = total_time / m


if __name__ == '__main__':
    target = u0_3
    grid1, hx, tau = create_grid(w, total_time, n, m, target, mu)
    grid2, hx, tau = create_grid(w, total_time, n, m, target, mu)
    
    c = abs(a) * tau / hx
    if c > 1:
        print('Не сходится: c = ', c)
        exit(1)
    
    print(f"Initial parameters:")
    print(f"h_x = {h_x:.6f}, h_t = {h_t:.6f}, CFL = {c:.4f}\n")
    
    grid1 = solve1(grid1, a, hx, tau)
    grid2 = solve2(grid2, a, hx, tau)
    
    x_vals = np.linspace(0, w, n)
    
    steps_per_second = int(1 / h_t)
    for second in range(1, total_time + 1):
        step = second * steps_per_second
        if step >= m:
            break
        precise_solution = generate_precise(target, a, step, hx, n, tau)
        print_comparison_table(step, x_vals, grid1[step], grid2[step], precise_solution, h_x, h_t)
    
    fig, ax = plt.subplots()
    x = np.linspace(0, w, n)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    def update(frame):
        plt.cla()
        current_time = frame * h_t
        ax.plot(x, grid1[frame], '-', color='orange', label=f'method 1 (time={current_time:.2f})')
        ax.plot(x, grid2[frame], '--', color='blue', label=f'method 2 (time={current_time:.2f})')
        ax.plot(x, generate_precise(target, a, frame, hx, n, tau), ':', color='red', label='precise')
        plt.legend()
        
        # Останавливаем анимацию на 3 секунде (frame = 3/h_t)
        if abs(current_time - 2.0) < h_t/2:
            plt.pause(10)  # Пауза на 2 секунды

    ani = animation.FuncAnimation(fig, update, frames=m, interval=30)
    plt.show()