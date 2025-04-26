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
    return float(x >= 1 and x <= 2)

def u0_1(x):
    if x < 1 or x > 2:
        return 0
    elif x >= 1 and x < 1.5:
        return 2 * (x - 1)
    else:
        return 1 - 2 * (x - 1.5)

def u0_2(x):
    if x < 1 or x > 2:
        return 0
    else:
        return 0.5 * (1 + sin(2 * pi * (x - 1) - pi / 2))

def mu(t):
    return 0

def solve1(grid, a, h, tau):
    for t in range(1, len(grid)):
        for x in range(1, len(grid[t])):
            grid[t][x] = grid[t - 1][x] - a * tau * ((grid[t - 1][x] - grid[t - 1][x - 1]) / h)
    return grid

# ЗАМЕНЁННЫЙ НА УСТОЙЧИВЫЙ МЕТОД (СХЕМА ЛАКСА)
def solve2(grid, a, h, tau):
    for t in range(1, len(grid)):
        for x in range(1, len(grid[t]) - 1):
            grid[t][x] = 0.5 * (grid[t - 1][x + 1] + grid[t - 1][x - 1]) \
                         - a * tau / (2 * h) * (grid[t - 1][x + 1] - grid[t - 1][x - 1])
    return grid

def solve_single(t, grid, a, h, tau):
    prev = grid[t - 1]
    curr = grid[t]

    ti   = 1 / tau

    mu1  = curr[0]
    mu2  = 0

    kappa1 = 0
    kappa2 = 0

    ai = [kappa1]
    bi = [mu1]

    C = -ti
    A = -a / (4 * h)
    B =  a / (4 * h)

    for x in range(1, len(curr) - 1):
        phi  = prev[x] * ti - a / (4 * h) * (prev[x + 1] - prev[x - 1])
        alpha = B / (C - A * ai[x - 1])
        beta  = (-phi + A * bi[x - 1]) / (C - A * ai[x - 1])
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

a = 2
w = 15
h = 10
n = w * 10
m = h * 20

# tau / h < 0.5

if __name__ == '__main__':
    target = u0_2

    grid1, hx, tau = create_grid(w, h, n, m, target, mu)
    grid2, hx, tau = create_grid(w, h, n, m, target, mu)
    grid3, hx, tau = create_grid(w, h, n, m, target, mu)

    c = abs(a) * tau / hx
    if c > 1:
        print('Doesnt converge: c = ', c)
        print(tau, hx)
        exit(1)

    grid1 = solve1(grid1, a, hx, tau)
    grid2 = solve2(grid2, a, hx, tau)
    grid3 = solve3(grid3, a, hx, tau)

    fig, ax = plt.subplots()
    x = np.linspace(0, w, n)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    def update(frame):
        plt.cla()
        ax.plot(x, grid1[frame], '-', color='yellow', label='Метод №1')
        ax.plot(x, grid2[frame], '--', color='blue', label='Метод №2')
        ax.plot(x, grid3[frame], '-.', color='green', label='Метод №3')
        ax.plot(x, generate_precise(target, a, frame, hx, n, tau), ':', color='red', label='Точное решение')
        plt.legend()

    ani = animation.FuncAnimation(fig, update, frames=m, interval=30)

    plt.show()
