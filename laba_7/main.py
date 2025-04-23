from math import sin, pi
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class InitialConditions:
    @staticmethod
    def u0_0(x):
        return float(1 <= x <= 2)

    @staticmethod
    def u0_1(x):
        if x < 1 or x > 2:
            return 0
        elif x < 1.5:
            return 2 * (x - 1)
        else:
            return 1 - 2 * (x - 1.5)

    @staticmethod
    def u0_2(x):
        if x < 1 or x > 2:
            return 0
        else:
            return 0.5 * (1 + sin(2 * pi * (x - 1) - pi / 2))

    @staticmethod
    def mu(t):
        return 0

class Grid:
    def __init__(self, length, time_slice, n, m, u0, mu):
        self.n = n
        self.m = m
        self.h = length / n
        self.tau = time_slice / m
        self.grid = self._create_grid(u0, mu)

    def _create_grid(self, u0, mu):
        grid = [[0] * self.n for _ in range(self.m)]
        for i in range(self.n):
            grid[0][i] = u0(i * self.h)
        for j in range(self.m):
            grid[j][0] = mu(j * self.tau)
        return grid

class Solver:
    def __init__(self, a):
        self.a = a

    def solve(self, grid_obj: Grid):
        grid = grid_obj.grid
        for t in range(1, grid_obj.m):
            self._solve_single(t, grid, self.a, grid_obj.h, grid_obj.tau)

    def _solve_single(self, t, grid, a, h, tau):
        prev = grid[t - 1]
        curr = grid[t]
        hs = 1 / h**2
        ti = 1 / tau
        ah = a / 2

        mu1 = curr[0]
        mu2 = 0
        kappa1 = 0
        kappa2 = 0

        ai = [kappa1]
        bi = [mu1]

        C = -(ti + a * hs)

        for x in range(1, len(curr) - 1):
            A = -0.25 / h * prev[x - 1] - ah * hs
            B = 0.25 / h * prev[x + 1] - ah * hs
            phi = prev[x] * ti + ah * hs * (prev[x + 1] - 2 * prev[x] + prev[x - 1])

            alpha = B / (C - A * ai[x - 1])
            beta = (-phi + A * bi[x - 1]) / (C - A * ai[x - 1])

            ai.append(alpha)
            bi.append(beta)

        curr[-1] = (kappa2 * bi[-1] + mu2) / (1 - kappa2 * ai[-1])
        for x in range(len(curr) - 2, -1, -1):
            curr[x] = ai[x] * curr[x + 1] + bi[x]

class Visualizer:
    def __init__(self, grid, width):
        self.grid = grid
        self.n = len(grid[0])
        self.m = len(grid)
        self.width = width

    def animate(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, self.width, self.n)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))

        def update(frame):
            plt.cla()
            ax.plot(x, self.grid[frame], '-')
            plt.legend()

        ani = animation.FuncAnimation(fig, update, frames=self.m, interval=30)
        plt.show()

if __name__ == '__main__':
    a = 0.05
    w = 15
    h = 10
    n = w * 10
    m = h * 20

    grid_obj = Grid(w, h, n, m, InitialConditions.u0_1, InitialConditions.mu)
    solver = Solver(a)
    solver.solve(grid_obj)

    vis = Visualizer(grid_obj.grid, w)
    vis.animate()
