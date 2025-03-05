import numpy as np
from typing import List, Tuple


class Zeidel:
    def __init__(
        self,
        A: List[List[float]],
        b: List[float],
        x_star: List[float],
        nmax: int,
        epsilon: float,
    ):
        self.A: np.ndarray = np.array(A, dtype=float)
        self.b: np.ndarray = np.array(b, dtype=float)
        self.x_star: np.ndarray = np.array(x_star, dtype=float)
        self.nmax: int = nmax
        self.epsilon: float = epsilon
        self.n: int = len(b)
        self.x: np.ndarray = np.ones(self.n, dtype=float)

    @staticmethod
    def norma(x: np.ndarray) -> float:
        """Вычисляет норму Чебышева (максимальное абсолютное значение вектора)."""
        return float(np.max(np.abs(x)))

    @staticmethod
    def accuracy(x_old: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Вычисляет разницу между текущим и предыдущим приближением."""
        return np.abs(x - x_old)

    def residual(self) -> np.ndarray:
        """Вычисляет вектор невязки Ax - b."""
        return np.dot(self.A, self.x) - self.b

    def check_matrix(self) -> bool:
        """Проверяет, является ли матрица симметричной и положительно определённой."""
        return np.allclose(self.A, self.A.T) and np.all(np.linalg.eigvals(self.A) > 0)

    def solve(
        self,
    ) -> Tuple[List[float], int, float, List[float], bool, List[float], float]:
        """Решает СЛАУ методом Зейделя."""
        iter_count: int = 0
        epsilon_reached: bool = False

        while iter_count < self.nmax and not epsilon_reached:
            x_old = self.x.copy()
            for i in range(self.n):
                sum1 = sum(self.A[i, j] * self.x[j] for j in range(i))
                sum2 = sum(self.A[i, j] * x_old[j] for j in range(i + 1, self.n))
                self.x[i] = (self.b[i] - sum1 - sum2) / self.A[i, i]

            error_vector = self.accuracy(x_old, self.x)
            error_norm = self.norma(error_vector)

            if error_norm <= self.epsilon:
                epsilon_reached = True

            iter_count += 1

        residual_vector = self.residual()
        return (
            self.x.tolist(),
            int(iter_count),
            float(error_norm),
            error_vector.tolist(),
            epsilon_reached,
            residual_vector.tolist(),
            float(self.norma(residual_vector)),
        )

    def run(self):
        """Основной метод для запуска алгоритма и вывода результатов."""
        if not self.check_matrix():
            print(
                "Ошибка: Матрица A должна быть симметричной и положительно определённой."
            )
            return

        (
            x,
            iterations,
            error_norm,
            error_vector,
            by_epsilon,
            residual_vector,
            residual_norm,
        ) = self.solve()
        exact_error_vector = self.accuracy(np.array(x, dtype=float), self.x_star)
        exact_error = self.norma(exact_error_vector)

        print("\nЧисленное решение:", x)
        print(f"Точность: {error_norm}, по компонентам: {error_vector}")
        print(
            f"Погрешность: {exact_error}, по компонентам: {exact_error_vector.tolist()}"
        )
        print(f"Количество итераций: {iterations}")
        print("Выход по точности" if by_epsilon else "Выход по числу итераций")
        print(f"Невязка: {residual_norm}")
        print(f"Вектор невязки: {residual_vector}")
