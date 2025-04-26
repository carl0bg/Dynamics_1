# Метод Зейделя

import math
from dataclasses import dataclass, field
import tabulate
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PoissonProblem:
    A: float = 5
    b: float = 0

    def mu(self, i, j, grid):
        return 0

    def f(self, i, j, grid):
        x = i * grid.h
        y = j * grid.k
        r = (x - 0.5) ** 2 + (y - 0.5) ** 2
        return self.A * math.exp(-self.b * r ** 2)

    def u_exact(self, x, y):
        return 0


@dataclass
class GridSolver:
    width: float
    height: float
    n: int
    m: int
    problem: PoissonProblem
    h: float = field(init=False)
    k: float = field(init=False)
    h_sq_inv: float = field(init=False)
    k_sq_inv: float = field(init=False)
    a_star: float = field(init=False)
    grid: np.ndarray = field(init=False)
    accuracy: list = field(init=False, default_factory=list)
    X: np.ndarray = field(init=False)
    Y: np.ndarray = field(init=False)
    uv: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.h = self.width / self.n
        self.k = self.height / self.m
        self.h_sq_inv = 1 / (self.h ** 2)
        self.k_sq_inv = 1 / (self.k ** 2)
        self.a_star = -2 * (self.h_sq_inv + self.k_sq_inv)

        self.grid = np.zeros((self.m + 1, self.n + 1))
        x = np.linspace(0, self.width, self.n + 1)
        y = np.linspace(0, self.height, self.m + 1)
        self.X, self.Y = np.meshgrid(x, y)

        if hasattr(self.problem, 'u_exact'):
            self.uv = np.array(self.problem.u_exact(np.ravel(self.X), np.ravel(self.Y)))

        self.__apply_boundary_conditions()

    def __apply_boundary_conditions(self):
        for j in range(self.m + 1):
            for i in range(self.n + 1):
                self.grid[j][i] = self.problem.mu(i, j, self)

    def __update(self, i, j):
        prev = self.grid[j][i]
        left = self.h_sq_inv * self.grid[j][i - 1]
        right = self.h_sq_inv * self.grid[j][i + 1]
        up = self.k_sq_inv * self.grid[j + 1][i]
        down = self.k_sq_inv * self.grid[j - 1][i]
        f_val = self.problem.f(i, j, self)
        new_val = (f_val + left + right + up + down) / -self.a_star
        self.grid[j][i] = new_val
        return abs(prev - new_val)

    def __opt_update(self, i, j):
        prev = self.grid[j][i]
        left = self.grid[j][i - 1]
        right = self.grid[j][i + 1]
        up = self.grid[j + 1][i]
        down = self.grid[j - 1][i]
        f_val = self.problem.f(i, j, self)
        new_val = 0.4 * (f_val * self.k ** 2 + 0.25 * (left + right) + (up + down))
        self.grid[j][i] = new_val
        return abs(prev - new_val)

    def solve(self, precision=1e-10, max_iterations=1000, optimized=False):
        update = self.__opt_update if optimized else self.__update
        max_delta = precision + 1
        iteration = 0
        self.accuracy = []

        while iteration < max_iterations and max_delta > precision:
            max_delta = 0
            for j in range(1, self.m):
                for i in range(1, self.n):
                    delta = update(i, j)
                    max_delta = max(max_delta, delta)
            iteration += 1
            self.accuracy.append(max_delta)
            print(f'Итерация {iteration}, z = {max_delta:.2e}')

        error = self.calculate_error() if self.uv is not None else None
        return iteration, max_delta, error

    def calculate_error(self):
        z = np.ravel(self.grid)
        diffs = np.abs(z - self.uv)
        return np.nanmax(diffs)

    def plot(self):
        fig, ax = plt.subplots()
        # Устанавливаем фиксированные vmin и vmax
        vmin = np.min(self.grid)
        vmax = np.max(self.grid)

        cs = ax.contourf(self.X, self.Y, self.grid, vmin=0, vmax=0.65)

        fig.colorbar(cs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.xaxis.set_major_locator(plt.MultipleLocator(self.h))
        ax.yaxis.set_major_locator(plt.MultipleLocator(self.k))
        plt.grid()
        plt.show()


if __name__ == '__main__':
    problem = PoissonProblem(A=5, b=3)
    solver = GridSolver(width=1, height=1, n=20, m=20, problem=problem)

    max_iter = 1000
    iters, acc, err = solver.solve(precision=1e-14, max_iterations=max_iter, optimized=True)

    print(tabulate.tabulate([
        ['Количество итераций', f'{iters}/{max_iter}'],
        ['Точность', acc],
        ['Погрешность', err]
    ], tablefmt='simple_grid', colalign=('left', 'right')))

    solver.plot()
