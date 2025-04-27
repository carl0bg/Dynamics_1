from math import sin, pi
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tabulate import tabulate


def create_grid(length, time_slice, n, m, u0, mu):
    grid = []

    h = length / n
    t = time_slice / m

    for j in range(m):
        grid.append([0] * n)

    for i in range(n):
        grid[0][i] = u0(i * h)

    for j in range(m):
        grid[j][0] = mu(j * t)

    return grid, h, t

def u0_0(x):
    return 1.0 if 1 <= x <= 2 else 0.0  # прямоугольный импульс

def u0_1(x):
    if x < 1 or x > 2:
        return 0
    elif x >= 1 and x < 1.5:
        return 2 * (x - 1)
    else:
        return 1 - 2 * (x - 1.5) # треугольный импульс

def u0_2(x):
    if x < 1 or x > 2:
        return 0
    else:
        return 0.5 * (1 + sin(2 * pi * (x - 1) - pi / 2)) # синусоидальный импульс
    

def u0_3(x):
    if x < 0 or x > 1:
        return 0
    else:
        return 1 - x**2  # функция для половины параболы

def mu(t):
    return 0

def solve_single(t, grid, nu, h, tau):
    """Неявная схема для уравнения Бюргерса"""
    prev = grid[t - 1]
    curr = grid[t]
    n = len(curr)
    
    # Коэффициенты схемы
    c1 = 1.0 / tau
    c2 = 1.0 / (2 * h)  # для конвективного члена
    c3 = nu / (h ** 2)  # для диффузионного члена
    
    # Прогоночные коэффициенты
    alpha = [0.0] * n
    beta = [0.0] * n
    
    # Левое граничное условие (u[0] задано)
    alpha[1] = 0.0
    beta[1] = curr[0]
    
    # Заполнение прогоночных коэффициентов
    for i in range(1, n - 1):
        u_val = prev[i]  # используем значение с предыдущего временного слоя
        
        A = u_val * c2 + c3
        B = -c1 - 2 * c3
        C = -u_val * c2 + c3
        F = -c1 * prev[i]
        
        alpha[i + 1] = -C / (B + A * alpha[i])
        beta[i + 1] = (F - A * beta[i]) / (B + A * alpha[i])
    
    # Правое граничное условие (нейтральное)
    curr[-1] = beta[-1] / (1 - alpha[-1])
    
    # Обратный ход прогонки
    for i in range(n - 2, -1, -1):
        curr[i] = alpha[i + 1] * curr[i + 1] + beta[i + 1]


def solve(grid, a, h, tau):
    for t in range(1, len(grid)):
        solve_single(t, grid, a, h, tau)

def print_solution_table(time_step, x_vals, solution, h_x, h_t):
    table_data = []
    headers = [
        "x (шаг h_x={:.4f})".format(h_x),
        "Значение u(x, t={:.2f})".format(time_step * h_t)
    ]
    
    for i in range(len(x_vals)):
        x = x_vals[i]
        val = solution[i]
        table_data.append([
            f"{x:.4f}",
            f"{val:.6f}"
        ])
    
    print(f"\n t = {time_step*h_t:.2f}, h_t = {h_t:.4f}:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))

v = 0.05 # к-ф вязкости
w = 15 # x
total_time = 10 # t
n = w * 12  # число пространственных узлов
m = total_time * 25  # число временных шагов
h_x = w / n
h_t = total_time / m

if __name__ == '__main__':
    target = u0_0

    grid, hx, tau = create_grid(w, total_time, n, m, target, mu)

    solve(grid, v, hx, tau)

    x_vals = np.linspace(0, w, n)

    # Запрос времени у пользователя
    while True:
        try:
            input_time = float(input("Введите время для вывода решения: "))
            if 0 <= input_time <= total_time:
                break
            else:
                print("Ошибка: время должно быть в диапазоне [0, {}]".format(total_time))
        except ValueError:
            print("Ошибка: введите числовое значение")

    # Находим ближайший временной шаг
    time_step = int(round(input_time / h_t))
    if time_step >= len(grid):
        time_step = len(grid) - 1

    # Выводим таблицу для выбранного времени
    print_solution_table(time_step, x_vals, grid[time_step], h_x, h_t)

    fig, ax = plt.subplots()
    x = np.linspace(0, w, n)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    def update(frame):
        plt.cla()
        plt.title(f"t = {tau * frame:.2f}")
        ax.plot(x, grid[frame], '-')
        plt.legend()

    if False:
        ani = animation.FuncAnimation(fig, update, frames=m, interval=30, repeat=False)
    else:
        update(time_step)

    plt.show()