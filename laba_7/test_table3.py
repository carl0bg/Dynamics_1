from math import sin, pi, fabs
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

def print_combined_solution_table(time_step, x_vals1, solution1, x_vals2, solution2, h_t):
    table_data = []
    headers = [
        "x",
        "U1 (n=180)",
        "U2 (n=90)",
        "Diff (U1-U2)"
    ]
    
    # Интерполируем решение с более грубой сетки на более точную
    interp_solution2 = np.interp(x_vals1, x_vals2, solution2)
    
    for i in range(len(x_vals1)):
        x = x_vals1[i]
        val1 = solution1[i]
        val2 = interp_solution2[i]
        diff = fabs(val1 - val2)
        
        table_data.append([
            f"{x:.4f}",
            f"{val1:.6f}",
            f"{val2:.6f}",
            f"{diff:.6f}"
        ])
    
    print(f"\nСравнение решений при t = {time_step*h_t:.2f}:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))

v = 0.05 # к-ф вязкости
w = 15 # x
total_time = 10 # t

if __name__ == '__main__':
    target = u0_0

    # Первая сетка (точная)
    n1 = w * 12
    m1 = total_time * 25
    grid1, hx1, tau1 = create_grid(w, total_time, n1, m1, target, mu)
    solve(grid1, v, hx1, tau1)
    x_vals1 = np.linspace(0, w, n1)

    # Вторая сетка (грубая)
    n2 = w * 6
    m2 = total_time * 25  # одинаковое количество временных шагов
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
                print(f"Ошибка: время должно быть в диапазоне [0, {total_time}]")
        except ValueError:
            print("Ошибка: введите числовое значение")

    # Используем одинаковый временной шаг для обоих сеток
    time_step = int(round(input_time / tau1))
    if time_step >= len(grid1):
        time_step = len(grid1) - 1

    # Выводим объединенную таблицу
    print_combined_solution_table(
        time_step, 
        x_vals1, 
        grid1[time_step], 
        x_vals2, 
        grid2[time_step], 
        tau1
    )

    # Визуализация
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    plt.title(f"Сравнение решений, t = {tau1 * time_step:.2f}")
    plt.plot(x_vals1, grid1[time_step], '-', label=f"Точная сетка (n={n1})")
    plt.plot(x_vals2, grid2[time_step], '--', label=f"Грубая сетка (n={n2})")
    plt.legend()
    plt.show()