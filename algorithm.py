import math


class ThomasAlgorithm:
    def __init__(self, n):
        self.n = n
        self.h = 1 / n
        self.mu1 = 10
        self.mu2 = 100
        self.kappa1 = self.kappa2 = 0
        self.A = 12 / (self.h ** 2)
        self.B = 12 / (self.h ** 2)
        self.C = 24 / (self.h ** 2) + 5

    @staticmethod
    def analysis(x):
        """Аналитическое решение."""
        return 10 + 90 * (x ** 2)

    @staticmethod
    def right_sade(x):
        """Правая часть уравнения."""
        return -2110 + 450 * (x ** 2)

    def solve(self):
        """Решение краевой задачи методом прогонки."""
        ai = [self.kappa1]
        bi = [self.mu1]
        xi = [0]

        for i in range(1, self.n):
            _xi = i * self.h
            xi.append(_xi)
            a = self.B / (self.C - self.A * ai[i - 1])
            b = (self.right_sade(_xi) + self.A * bi[i - 1]) / (self.C - self.A * ai[i - 1])
            ai.append(a)
            bi.append(b)

        xi.append(self.n * self.h)

        yn = (self.kappa2 * bi[self.n - 1] + self.mu2) / (1 - self.kappa2 * ai[self.n - 1])
        yi = [yn]

        for i in range(self.n - 1, -1, -1):
            yi.append(ai[i] * yi[self.n - i - 1] + bi[i])

        return xi, yi[::-1]

    @classmethod
    def compute_error(cls, xs, ys):
        """Максимальное отклонение численного решения от аналитического."""
        return max(math.fabs(cls.analysis(xs[i]) - ys[i]) for i in range(len(xs)))