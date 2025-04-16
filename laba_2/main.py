from utils import read_input_file
from zeidel import Zeidel

# test
if __name__ == "__main__":
    nmax = int(input("Введите максимальное число итераций (nmax): "))
    epsilon = float(input("Введите требуемую точность (epsilon): "))

    A, b, x_star = read_input_file("input.txt")

    solver = Zeidel(A, b, x_star, nmax, epsilon)
    solver.run()
