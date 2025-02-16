from solver import Solver


if __name__ == '__main__':
    s = input('Enter grid size: ')

    if s.strip():
        Solver.run_with_fixed_n(int(s))
    else:
        Solver.run_with_incremental_n()