import numpy as np
from matplotlib import pyplot as plt

from algorithm import ThomasAlgorithm
from drawplot import DrawPlot


class Solver:
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
    def run_with_incremental_n():
        n = 10
        while n <= 10000000:
            problem = ThomasAlgorithm(n)
            vxs, vys = problem.solve()
            print(f"n = {n}, z = {ThomasAlgorithm.compute_error(vxs, vys)}")
            n *= 10
