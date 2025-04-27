from math import sin, pi
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


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

a = 0.05 # к-ф вязкости
w = 15 # x
total_time = 10 # t
n = w * 10  # число пространственных узлов
m = total_time * 20  # число временных шагов


if __name__ == '__main__':

    target = u0_0

    grid, hx, tau = create_grid(w, total_time, n, m, target, mu)

    solve(grid, a, hx, tau)

    fig, ax = plt.subplots()
    x = np.linspace(0, w, n)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    def update(frame):
        plt.cla()
        plt.title(f"t = {tau * frame:.2f}")
        ax.plot(x, grid[frame], '-')
        plt.legend()

        print(f't = {frame}')
        print(grid[frame])

    fr = input('Time: ')
    anim = len(fr) == 0
    if not anim:
        fr = int(float(fr) / tau)

    if anim:
        ani = animation.FuncAnimation(fig, update, frames=m, interval=30, repeat = False)
    else:
        update(fr)

    plt.show()