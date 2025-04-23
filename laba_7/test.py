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

def solve_single(t, grid, a, h, tau):
    prev = grid[t - 1]
    curr = grid[t]
    hs   = 1 / h**2
    ti   = 1 / tau
    ah   = a / 2
    mu1  = curr[0]
    mu2  = 0
    kappa1 = 0
    kappa2 = 0
    ai = [kappa1]
    bi = [mu1]
    C = -(ti + a * hs)
    for x in range(1, len(curr) - 1):
        A = -0.25 / h * prev[x - 1] - ah * hs
        B = 0.25 / h * prev[x + 1] - ah * hs
        phi  = prev[x] * ti + ah * hs * (prev[x + 1] - 2 * prev[x] + prev[x - 1])

        alpha = B / (C - A * ai[x - 1])
        beta  = (-phi + A * bi[x - 1]) / (C - A * ai[x - 1])

        ai.append(alpha)
        bi.append(beta)

    curr[-1] = (kappa2 * bi[-1] + mu2) / (1 - kappa2 * ai[-1])
    for x in range(len(curr) - 2, -1, -1):
        curr[x] = ai[x] * curr[x + 1] + bi[x]


def solve(grid, a, h, tau):
    for t in range(1, len(grid)):
        solve_single(t, grid, a, h, tau)

a = 0.05
w = 15
h = 10
n = w * 10
m = h * 20

if __name__ == '__main__':
    target = u0_1

    grid, hx, tau = create_grid(w, h, n, m, target, mu)

    solve(grid, a, hx, tau)

    fig, ax = plt.subplots()
    x = np.linspace(0, w, n)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    def update(frame):
        plt.cla()
        ax.plot(x, grid[frame], '-')
        plt.legend()

    ani = animation.FuncAnimation(fig, update, frames=m, interval=30)

    plt.show()