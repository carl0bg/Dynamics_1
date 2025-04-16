# Метод Сопряженных Градиентов

import math
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import tabulate


def mu(i, j, grid):
    x = i * grid.h
    y = j * grid.k

    if i == 0 and j >= grid.m // 4:
        return y ** 2
    if i == grid.n:
        return y ** 2 + 4
    if j == grid.m // 4 and i <= grid.n // 2:
        return x ** 2 + 0.0625
    if j == grid.m:
        return x ** 2 + 1
    if i == grid.n // 2 and j < grid.m // 4:
        return y ** 2 + 1
    if j == 0 and i >= grid.n // 2:
        return x ** 2
    if j < grid.m // 4 and i < grid.n // 2:
        return np.nan

    return 0


def f(i, j, grid):
    return -4


def u(x, y):
    return x**2 + y**2


@dataclass
class GridParams:
    width: float
    height: float
    n: int
    m: int
    f: callable
    mu: callable
    u: callable = None

    def __post_init__(self):
        self.h = self.width / self.n
        self.k = self.height / self.m
        self.k_sq = self.k ** 2
        self.h_star = 1 / self.h ** 2
        self.k_star = 1 / self.k_sq
        self.a_star = -2 * (self.h_star + self.k_star)


@dataclass
class GridSolver:
    params: GridParams
    grid: list = field(init=False)
    r: list = field(init=False)
    z: list = field(init=False)
    H: list = field(init=False)
    alpha: float = field(default=0)
    c: float = field(default=0)
    d: float = field(default=0)
    accuracy: list = field(default_factory=list)
    uv: np.ndarray = field(default=None)
    X: np.ndarray = field(init=False)
    Y: np.ndarray = field(init=False)

    def __post_init__(self):
        p = self.params
        self.grid = [[0.0] * (p.n + 1) for _ in range(p.m + 1)]
        self.r = [[0.0] * (p.n + 1) for _ in range(p.m + 1)]
        self.z = [[0.0] * (p.n + 1) for _ in range(p.m + 1)]
        self.H = [[0.0] * (p.n + 1) for _ in range(p.m + 1)]

        x = np.linspace(0, p.width, p.n + 1)
        y = np.linspace(0, p.height, p.m + 1)
        self.X, self.Y = np.meshgrid(x, y)

        if p.u:
            self.uv = np.array(p.u(np.ravel(self.X), np.ravel(self.Y)))

        for j in range(p.m + 1):
            for i in range(p.n + 1):
                self.set(i, j, p.mu(i, j, p))

    def get(self, i, j):
        return self.grid[j][i]

    def set(self, i, j, value):
        self.grid[j][i] = value

    def __update(self, i, j):
        prev = self.get(i, j)
        new = prev + self.alpha * self.H[j][i]
        self.set(i, j, new)
        return abs(prev - new)

    def __grid_cycle(self, func):
        p = self.params
        for j in range(1, p.m // 4 + 1):
            for i in range(p.n // 2 + 1, p.n):
                func(i, j)
        for j in range(p.m // 4 + 1, p.m):
            for i in range(1, p.n):
                func(i, j)

    def __calc_r(self, i, j):
        p = self.params
        prev = self.get(i, j)
        left = p.h_star * self.get(i - 1, j)
        right = p.h_star * self.get(i + 1, j)
        up = p.k_star * self.get(i, j + 1)
        down = p.k_star * self.get(i, j - 1)
        f_val = p.f(i, j, p)
        self.r[j][i] = f_val + p.a_star * prev + left + right + up + down
        self.H[j][i] = -self.r[j][i]

    def __calc_z(self, i, j):
        p = self.params
        prev = self.r[j][i]
        left = p.h_star * self.r[j][i - 1]
        right = p.h_star * self.r[j][i + 1]
        up = p.k_star * self.r[j + 1][i]
        down = p.k_star * self.r[j - 1][i]
        self.z[j][i] = p.a_star * prev + left + right + up + down

    def __calc_alpha(self, i, j):
        if i == 1 and j == 1:
            self.c = self.d = 0
        self.c += self.r[j][i] ** 2
        self.d += self.r[j][i] * self.z[j][i]

    def solve(self, precision=1e-14, max_iterations=10000):
        p = self.params
        max_delta = precision + 1
        iterations = 0

        while iterations < max_iterations and max_delta > precision:
            self.__grid_cycle(self.__calc_r)
            self.__grid_cycle(self.__calc_z)
            self.__grid_cycle(self.__calc_alpha)
            self.alpha = self.c / self.d
            max_delta = -1

            for j in range(1, p.m // 4 + 1):
                for i in range(p.n // 2 + 1, p.n):
                    d = self.__update(i, j)
                    max_delta = max(max_delta, d)
            for j in range(p.m // 4 + 1, p.m):
                for i in range(1, p.n):
                    d = self.__update(i, j)
                    max_delta = max(max_delta, d)

            iterations += 1
            self.accuracy.append(max_delta)
            print(f'Итерация: {iterations}, z = {max_delta}')

        error = None
        if self.uv is not None:
            z = np.ravel(self.grid)
            diffs = np.abs(z - self.uv)
            error = np.nanmax(diffs)

        return iterations, max_delta, error

    def plot(self):
        z = np.ravel(self.grid).reshape(self.X.shape)

        fig, ax = plt.subplots()
        cs = ax.contourf(self.X, self.Y, z)
        fig.colorbar(cs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.xaxis.set_major_locator(plt.MultipleLocator(self.params.h))
        ax.yaxis.set_major_locator(plt.MultipleLocator(self.params.k))
        plt.grid()

        rect = plt.Rectangle((0, 0), 1, 0.25, linewidth=1, edgecolor='none', facecolor='black')
        ax.add_patch(rect)

        plt.show()


if __name__ == '__main__':
    params = GridParams(2, 1, 20, 20, f=f, mu=mu, u=u)
    solver = GridSolver(params)
    iterations, acc, error = solver.solve(precision=1e-14, max_iterations=10000)
    print(tabulate.tabulate([
        ['Количество итераций', f'{iterations}/10000'],
        ['Точность', acc],
        ['Погрешность', error]
    ], tablefmt='simple_grid', colalign=('left', 'right')))
    solver.plot()
