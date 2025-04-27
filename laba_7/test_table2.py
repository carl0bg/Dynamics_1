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

if __name__ == '__main__':
    target = u0_0

    # Первая сетка (исходные параметры)
    n1 = w * 12
    m1 = total_time * 25
    grid1, hx1, tau1 = create_grid(w, total_time, n1, m1, target, mu)
    solve(grid1, v, hx1, tau1)
    x_vals1 = np.linspace(0, w, n1)

    # Вторая сетка (другие параметры)
    n2 = w * 6  # в 2 раза меньше узлов по пространству
    m2 = int(total_time * 12.5)  # в 2 раза меньше шагов по времени
    grid2, hx2, tau2 = create_grid(w, total_time, n2, m2, target, mu)
    solve(grid2, v, hx2, tau2)
    x_vals2 = np.linspace(0, w, n2)

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

    time_step1 = int(round(input_time / tau1))
    if time_step1 >= len(grid1):
        time_step1 = len(grid1) - 1
        
    time_step2 = int(round(input_time / tau2))
    if time_step2 >= len(grid2):
        time_step2 = len(grid2) - 1

    print(time_step1)
    print(time_step2)

    # Выводим таблицы для выбранного времени
    print("\nРешение на первой сетке (n={}, m={}):".format(n1, m1))
    print_solution_table(time_step1, x_vals1, grid1[time_step1], hx1, tau1)
    
    print("\nРешение на второй сетке (n={}, m={}):".format(n2, m2))
    print_solution_table(time_step2, x_vals2, grid2[time_step2], hx2, tau2)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    # Функция для анимации
    def update(frame):
        plt.cla()
        plt.title(f"Сравнение решений, t = {tau1 * frame:.2f}")
        
        # Рассчитываем соответствующий кадр для второй сетки
        frame2 = int(frame * (m2 / m1))
        if frame2 >= len(grid2):
            frame2 = len(grid2) - 1
            
        ax.plot(x_vals1, grid1[frame], '-', label=f"Сетка 1 (n={n1}, m={m1})")
        ax.plot(x_vals2, grid2[frame2], '--', label=f"Сетка 2 (n={n2}, m={m2})")
        plt.legend()
        # plt.grid()

    # Создаем анимацию
    if False:
        ani = animation.FuncAnimation(fig, update, frames=m1, interval=30, repeat=False)
    else:
        update(time_step2)
    plt.show()