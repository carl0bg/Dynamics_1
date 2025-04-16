from dataclasses import dataclass, field
from typing import List, Tuple
import math

import numpy as np
from matplotlib import pyplot as plt

from algorithm import ThomasAlgorithm
from drawplot import DrawPlot

@dataclass
class Solver:
    n: int
    mu1: float = 10.0
    mu2: float = 100.0
    kappa1: float = 0.0
    kappa2: float = 0.0
    h: float = field(init=False)

    def __post_init__(self) -> None:
        self.h = 1 / self.n

    @staticmethod
    def run_with_fixed_n(n: int, flg_draw: bool = False):
        problem = ThomasAlgorithm(n)
        uxs = np.linspace(0, 1, n)
        uys = problem.analysis(uxs)
        vxs, vys = problem.solve()

        z = ThomasAlgorithm.compute_error(vxs, vys)

        # Рисуем два графика на одной координатной оси
        if flg_draw:
            print(f"z = {z}")

            DrawPlot.plot_one(
                uxs, uys, "U(x) - Точное решение", color="b", style="-"
            )  # Синий сплошной
            DrawPlot.plot_one(
                vxs, vys, "V(x) - Приближённое решение", color="r", style="--", z=z, n=n
            )  # Красный пунктирный

            plt.xlabel("x")
            plt.ylabel("y")
            plt.yticks(np.arange(0, 110, 10))
            plt.grid()
            plt.legend()
            plt.show()
        else:
            print(f"Итераций n = {n}, z = {z}")

    @staticmethod
    def u(x: float) -> float:
        return 10 + 90 * (x**2)

    @staticmethod
    def phi(x: float) -> float:
        return -2110 + 450 * (x**2)

    def solve(self) -> Tuple[List[float], List[float]]:
        A = 12 / (self.h**2)
        B = 12 / (self.h**2)
        C = 24 / (self.h**2) + 5

        ai = [self.kappa1]
        bi = [self.mu1]
        xi = [0.0]

        for i in range(1, self.n):
            _xi = i * self.h
            xi.append(_xi)
            a = B / (C - A * ai[i - 1])
            b = (self.phi(_xi) + A * bi[i - 1]) / (C - A * ai[i - 1])
            ai.append(a)
            bi.append(b)

        xi.append(self.n * self.h)

        yn = (self.kappa2 * bi[self.n - 1] + self.mu2) / (
            1 - self.kappa2 * ai[self.n - 1]
        )
        yi = [yn]

        for i in range(self.n - 1, -1, -1):
            yi.append(ai[i] * yi[self.n - i - 1] + bi[i])

        return xi, yi[::-1]


@dataclass
class OptimizedSolver(Solver):
    def solve(self, optimize_a: bool) -> Tuple[List[float], List[float]]:
        hs = self.h**2 / 12
        A = 12 / (self.h**2)
        B = 12 / (self.h**2)
        C = 24 / (self.h**2) + 5

        ai = [self.kappa1]
        bi = [self.mu1]
        xi = [0.0]

        for i in range(1, self.n):
            _xi = i * self.h
            xi.append(_xi)

            if optimize_a:
                a = 1 / ((hs * 5 - ai[i - 1]) + 2)
            else:
                a = B / (C - A * ai[i - 1])

            b = (hs * self.phi(_xi) + bi[i - 1]) * a
            ai.append(a)
            bi.append(b)

        xi.append(self.n * self.h)
        yn = self.mu2
        yi = [yn]

        for i in range(self.n - 1, -1, -1):
            yi.append(ai[i] * yi[self.n - i - 1] + bi[i])

        return xi, yi[::-1]


def compute_error(xs: List[float], ys: List[float]) -> float:
    return max(math.fabs(Solver.u(x) - y) for x, y in zip(xs, ys))
