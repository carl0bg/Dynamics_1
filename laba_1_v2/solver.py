from dataclasses import dataclass, field
from typing import List, Tuple
import math


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
