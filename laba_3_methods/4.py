# Метод Зейделя

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import tabulate


def mu(i, j, h, k):
    x = i * h
    y = j * k
    if i > 6 and j < 2:
        return 0
    if i == 0:
        return y
    if i == 8:
        return 2 * (y - 1)
    if j == 4:
        return 2
    return 0


@dataclass
class GridParams:
    width: float
    height: float
    n: int
    m: int
    precision: float = 1e-14
    max_iterations: int = 1000

    def __post_init__(self):
        self.h = self.width / self.n
        self.k = self.height / self.m
        self.h_star = 1 / self.h**2
        self.k_star = 1 / self.k**2
        self.a_star = -2 * (self.h_star + self.k_star)


@dataclass
class SeidelSolver:
    params: GridParams
    grid: list = field(init=False)
    bad_nodes: list = field(default_factory=lambda: [(6, 1), (7, 2)])
    invalid_nodes: list = field(default_factory=lambda: [(7, 1)])
    accuracy: list = field(default_factory=list)

    def __post_init__(self):
        p = self.params
        self.grid = [[mu(i, j, p.h, p.k) for i in range(p.n + 1)] for j in range(p.m + 1)]

    def solve(self):
        p = self.params
        max_delta = p.precision + 1
        iterations = 0

        while iterations < p.max_iterations and max_delta > p.precision:
            max_delta = -1
            for j in range(1, p.m):
                for i in range(1, p.n):
                    if (i, j) in self.invalid_nodes:
                        continue

                    left = p.h_star * self.grid[j][i - 1]
                    right = p.h_star * self.grid[j][i + 1]
                    up = p.k_star * self.grid[j + 1][i]
                    down = p.k_star * self.grid[j - 1][i]
                    f = 0
                    old = self.grid[j][i]

                    if (i, j) in self.bad_nodes:
                        if i == 6:
                            right = 0
                            xw = math.sqrt(1 - (j * p.k) ** 2) + 4
                            x = i * p.h
                            alpha = (xw - x) / p.h
                            a_star_corr = -2 * (p.k_star + p.h_star / alpha)
                            self.grid[j][i] = (up + down + 2 * (left / (1 + alpha) +
                                                               right / (alpha * (1 + alpha)))) / -a_star_corr
                        else:
                            down = 0
                            yw = math.sqrt(1 - (i * p.h - 4) ** 2)
                            y = j * p.k
                            beta = (y - yw) / p.k
                            a_star_corr = -2 * (p.h_star + p.k_star / beta)
                            self.grid[j][i] = (left + right + 2 * (up / (1 + beta) +
                                                                  down / (beta * (1 + beta)))) / -a_star_corr
                    else:
                        self.grid[j][i] = (f + left + right + up + down) / -p.a_star

                    d = abs(self.grid[j][i] - old)
                    if d > max_delta:
                        max_delta = d

            iterations += 1
            self.accuracy.append(max_delta)
            print(f"Итерация: {iterations}, z = {max_delta}")

        return iterations, max_delta

    def plot(self):
        p = self.params
        x = np.linspace(0, p.width, p.n + 1)
        y = np.linspace(0, p.height, p.m + 1)
        X, Y = np.meshgrid(x, y)
        Z = np.array(self.grid)

        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, Z)
        fig.colorbar(cs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim((0, p.width))
        ax.set_ylim((0, p.height))
        ax.xaxis.set_major_locator(plt.MultipleLocator(p.h))
        ax.yaxis.set_major_locator(plt.MultipleLocator(p.k))

        plt.grid()
        circle = plt.Circle((4, 0), 1, color='black', clip_on=True)
        ax.add_patch(circle)

        plt.show()


if __name__ == '__main__':
    params = GridParams(width=4, height=2, n=8, m=4)
    solver = SeidelSolver(params)
    iterations, final_accuracy = solver.solve()
    print(tabulate.tabulate([
        ['Количество итераций', f'{iterations}/{params.max_iterations}'],
        ['Точность', final_accuracy],
    ], tablefmt='simple_grid', colalign=('left', 'right')))
    solver.plot()
