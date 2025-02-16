import numpy as np
from matplotlib import pyplot as plt

from algorithm import ThomasAlgorithm
from drawplot import DrawPlot


class Solver:
    @staticmethod
    def run_with_fixed_n(n: int):
        problem = ThomasAlgorithm(n)
        uxs = np.linspace(0, 1, n)
        uys = problem.analysis(uxs)
        vxs, vys = problem.solve()
        print(f'z = {ThomasAlgorithm.compute_error(vxs, vys)}')
        DrawPlot.plot(1, uxs, uys, 'U(x)')
        DrawPlot.plot(2, vxs, vys, 'V(x)')
        plt.show()

    @staticmethod
    def run_with_incremental_n():
        n = 10
        while n <= 10000000:
            problem = ThomasAlgorithm(n)
            vxs, vys = problem.solve()
            print(f'n = {n}, z = {ThomasAlgorithm.compute_error(vxs, vys)}')
            n *= 10