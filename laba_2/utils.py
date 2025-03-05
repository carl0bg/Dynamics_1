from typing import List, Tuple


def read_input_file(
    filename: str,
) -> Tuple[List[List[float]], List[float], List[float]]:
    """Считывает матрицу A, вектор b и точное решение x_star из файла."""
    with open(filename, "r") as file:
        n = int(file.readline().strip())
        A, b = [], []
        for _ in range(n):
            row = list(map(float, file.readline().split()))
            A.append(row[:-1])
            b.append(row[-1])
        x_star = list(map(float, file.readline().split()))
    return A, b, x_star
