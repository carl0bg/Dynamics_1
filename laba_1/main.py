from solver import Solver


if __name__ == "__main__":
    s = input("Количество итераций: ")

    if s.strip():
        Solver.run_with_fixed_n(int(s), True)
    else:
        Solver.run_with_incremental_n()

    # Solver.run_with_fixed_n(10)
    # Solver.run_with_fixed_n(100)
    # Solver.run_with_fixed_n(1000)
    # Solver.run_with_fixed_n(10000)
    # Solver.run_with_fixed_n(100000)
