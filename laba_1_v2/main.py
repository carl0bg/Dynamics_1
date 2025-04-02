from solver import Solver, OptimizedSolver, compute_error
from plotter import Plotter
import numpy as np


def main():
    s = input("Введите размер сетки: ")

    if s:
        n = int(s)
        solver = Solver(n)
        opt_solver = OptimizedSolver(n)

        uxs = np.linspace(0, 1, n)
        uys = [solver.u(x) for x in uxs]

        vxs, vys = solver.solve()
        print(f"z = {compute_error(vxs, vys)}")

        vxs_opt1, vys_opt1 = opt_solver.solve(False)
        vxs_opt2, vys_opt2 = opt_solver.solve(True)

        print(
            f"z_opt1 = {compute_error(vxs_opt1, vys_opt1)}, z_opt2 = {compute_error(vxs_opt2, vys_opt2)}"
        )

        Plotter.make_plot(1, uxs, uys, "U(x)")
        Plotter.make_plot(2, vxs, vys, "V(x)")
        Plotter.make_plot(3, vxs_opt1, vys_opt1, "V(x) (Opt 1)")
        Plotter.make_plot(4, vxs_opt2, vys_opt2, "V(x) (Opt 2)")
        Plotter.show()


if __name__ == "__main__":
    main()
