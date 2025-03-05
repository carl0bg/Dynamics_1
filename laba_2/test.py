import numpy as np
import math


# Функция для вычисления нормы
def norm(x):
    return max(map(math.fabs, x))


# Функция для вычисления точности
def accuracy(x_old, x):
    errors = []
    for i in range(len(x)):
        errors.append(math.fabs(x[i] - x_old[i]))
    return errors


# Функция для вычисления погрешности (по сути делает то же самое, но для наглядности)
def error(x, x_star):
    return accuracy(x, x_star)


# Функция для вычисления вектора невязки
def res(A, x, b):
    return (np.matrix(A) * np.matrix(x).T - np.matrix(b).T).ravel().tolist()[0]


# Решение системы методом Зейделя
def solve_zeidel(A, b, nmax, e):
    n = len(b)
    x = [1] * n
    iter = 0
    e_cur = 0
    e_vec = []
    e_flag = False
    while iter < nmax and not e_flag:
        x_old = x.copy()
        # Непосредственно итерация происходит внутри этого цикла
        for i in range(n):
            x[i] = b[i] / A[i][i]
            for j in range(n):
                if i == j:
                    continue
                x[i] -= A[i][j] * x[j] / A[i][i]
        e_vec = accuracy(x_old, x)
        e_cur = norm(e_vec)
        if e_cur <= e:
            e_flag = True
        iter += 1
    r = res(A, x, b)
    return x, iter, e_cur, e_vec, e_flag, r, norm(r)


# Проверка симметричности и положительной определённости
def check_matrix(A):
    x = np.matrix(A)
    return np.allclose(x, x.T) and np.all(np.linalg.eigvals(x) > 0)


if __name__ == "__main__":
    nmax = int(input("Enter nmax: "))
    e = float(input("Enter e: "))
    n = 0
    A = []
    b = []
    x_star = []
    # Разбор файла
    with open("input.txt", "r") as file:
        n = int(file.readline())
        for i in range(n):
            row = list(map(float, file.readline().split(" ")))
            A.append(row[:3])
            b.append(int(row[3]))
        x_star = list(map(float, file.readline().split(" ")))
    if not check_matrix(A):
        print("Invalid matrix.")
        exit(1)
    x, ni, se, ev, by_e, r_vec, r = solve_zeidel(A, b, nmax, e)
    err_vec = error(x, x_star)
    err = norm(err_vec)

    print()
    print(f"Численное решение: {x}")
    print(f"Точность: {se}, по компонентам: {ev}")
    print(f"Погрешность: {err}, по компонентам: {err_vec}")
    print(f"Количество итераций: {ni}")
    if by_e:
        print(f"Выход по точности")
    else:
        print(f"Выход по числу итераций")
    print(f"Невязка: {r}")
    print(f"Вектор невязки: {r_vec}")
