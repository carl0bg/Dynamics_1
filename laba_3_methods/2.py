import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
from tabulate import tabulate
from math import pi

# Граничные условия (cos(pi k x) * cos(pi k y))
def mu(i, j, grid):
    x = i * grid.h
    y = j * grid.k
    if i == 0 or i == grid.n or j == 0 or j == grid.m:
        return math.cos(pi * grid._k * x) * math.cos(pi * grid._k * y)
    return 0

# Правая часть уравнения (нулевая, уравнение Лапласа)
def f(i, j, grid):
    return 0

# Аналитическое решение
def u(x, y, k=1):
    # return 0
    # return np.cos(pi * k * x) * np.cos(pi * k * y)
    return 0


class Grid:
    """
    Класс для решения уравнения Лапласа методом верхней релаксации.
    """

    def __init__(self, 
                 width: float, 
                 height: float, 
                 n: int, 
                 m: int, 
                 f: Callable, 
                 mu: Callable, 
                 w: float, 
                 k: int, 
                 exact_solution: Optional[Callable] = None):
        """
        Инициализация сетки.
        
        width: Ширина области
        height: Высота области
        n: Количество узлов по x
        m: Количество узлов по y
        f: Функция правой части уравнения
        mu: Функция граничных условий
        w: Параметр релаксации
        k: Параметр граничных условий
        exact_solution: Точное решение
        """
        self.width = width
        self.height = height
        self.n = n
        self.m = m
        self.w = w
        self._k = k

        # Шаги сетки
        self.h = width / n
        self.k_step = height / m

        # Предварительные вычисления для ускорения расчетов
        self.k_sq = self.k_step ** 2
        self.h_star = 1 / (self.h ** 2)
        self.k_star = 1 / self.k_sq
        self.a_star = 2 * (self.h_star + self.k_star)

        self.f = f
        self.exact_solution = exact_solution
        self.accuracy = []

        # инициализация сетки
        self._initialize_grid(mu)

        # Создание координатных сеток для визуализации
        self.x_coords = np.linspace(0, self.width, self.n + 1)
        self.y_coords = np.linspace(0, self.height, self.m + 1)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        
        # Точное решение (если предоставлено)
        if exact_solution:
            self.exact_values = exact_solution(np.ravel(self.X), np.ravel(self.Y), self._k)

    def _initialize_grid(self, mu: Callable) -> None:
        """Инициализирует сетку с граничными условиями."""
        self.grid = []
        for j in range(self.m + 1):
            self.grid.append([0.0] * (self.n + 1))
            for i in range(self.n + 1):
                self.set(i, j, mu(i, j, self))

    def get(self, i: int, j: int) -> float:
        """Возвращает значение в узле (i, j)."""
        return self.grid[j][i]

    def set(self, i: int, j: int, value: float) -> None:
        """Устанавливает значение в узле (i, j)."""
        self.grid[j][i] = value

    def _update_node(self, i: int, j: int) -> float:
        """Обновляет значение в узле (i, j) и возвращает изменение."""
        prev_value = self.get(i, j)
        
        # Соседние значения
        left = self.h_star * self.get(i - 1, j)
        right = self.h_star * self.get(i + 1, j)
        up = self.k_star * self.get(i, j + 1)
        down = self.k_star * self.get(i, j - 1)
        
        # Вычисление нового значения
        f_value = self.f(i, j, self)
        new_value = self.w * (left + right + up + down)
        new_value += (1 - self.w) * self.a_star * prev_value + self.w * f_value
        new_value /= self.a_star
        
        self.set(i, j, new_value)
        return abs(prev_value - new_value)

    def solve(self, precision: float, max_iterations: int) -> Tuple[int, float, Optional[float]]:
        """
        Решает уравнение методом релаксации.

        precision: Требуемая точность
        max_iterations: Максимальное число итераций
        """
        self.accuracy = []
        max_delta = precision + 1
        iterations = 0

        while iterations < max_iterations and max_delta > precision:
            max_delta = 0.0
            
            # Обход внутренних узлов
            for j in range(1, self.m):
                for i in range(1, self.n):
                    delta = self._update_node(i, j)
                    max_delta = max(max_delta, delta)
            
            iterations += 1
            self.accuracy.append(max_delta)
            print(f'Итерация: {iterations}, Δ = {max_delta:.2e}')

        # Вычисление погрешности (если есть точное решение)
        error = None
        if self.exact_solution:
            numeric_values = np.ravel(self.grid)
            diffs = np.abs(numeric_values - self.exact_values)
            error = np.max(diffs[~np.isnan(diffs)])

        return iterations, max_delta, error
    
    def print_final_table(self) -> None:
        """Печатает таблицу с результатами."""
        table = []
        for j in range(self.m + 1):
            for i in range(self.n + 1):
                x = i * self.h
                y = j * self.k_step
                v_numeric = self.get(i, j)
                
                row = [f"{x:.5f}", f"{y:.5f}", f"{v_numeric:.5e}"]
                
                if self.exact_solution:
                    v_exact = self.exact_solution(x, y, self._k)
                    error = abs(v_numeric - v_exact)
                    row.extend([f"{v_exact:.5e}", f"{error:.5e}"])
                
                table.append(row)

        headers = ["x", "y", "Численное"]
        if self.exact_solution:
            headers.extend(["Точное", "Погрешность"])
            
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def plot_solution(self) -> None:
        """Визуализирует решение."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        z = np.ravel(self.grid)
        Z = z.reshape(self.X.shape)
        
        # Контурный график
        cs = ax.contourf(self.X, self.Y, Z, levels=20, cmap='viridis')
        fig.colorbar(cs, ax=ax, label='Значение функции')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Численное решение уравнения Лапласа')
        
        # Сетка с шагом, кратным h и k_step
        ax.xaxis.set_major_locator(plt.MultipleLocator(self.h))
        ax.yaxis.set_major_locator(plt.MultipleLocator(self.k_step))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

def boundary_condition(i: int, j: int, grid: 'Grid') -> float:
    """Граничное условие: cos(πk x) * cos(πk y)."""
    x = i * grid.h
    y = j * grid.k_step
    if i == 0 or i == grid.n or j == 0 or j == grid.m:
        return math.cos(pi * grid._k * x) * math.cos(pi * grid._k * y)
    return 0.0


def zero_rhs(i: int, j: int, grid: 'Grid') -> float:
    """Правая часть уравнения (нулевая для уравнения Лапласа)."""
    return 0.0


def exact_solution(x: float, y: float, k: int = 1) -> float:
    """Точное решение (в данном случае нулевое)."""
    return 0.0


def main():
    """Основная функция для демонстрации работы метода."""
    # Параметры задачи
    k_param = 4
    grid_size = 20
    relaxation_param = 1.9
    max_iterations = 10000
    target_precision = 1e-14
    
    # Создание и решение сетки
    grid = Grid(
        width=1.0,
        height=1.0,
        n=grid_size,
        m=grid_size,
        f=zero_rhs,
        mu=boundary_condition,
        w=relaxation_param,
        k=k_param,
        exact_solution=exact_solution
    )
    
    iterations, accuracy, error = grid.solve(target_precision, max_iterations)
    
    # Вывод результатов
    grid.print_final_table()
    
    summary = [
        ['Количество итераций', f'{iterations}/{max_iterations}'],
        ['Достигнутая точность', f"{accuracy:.2e}"],
        ['Максимальная погрешность', f"{error:.2e}" if error is not None else "N/A"]
    ]
    
    print(tabulate(summary, tablefmt='simple_grid', colalign=('left', 'right')))
    
    # Визуализация
    grid.plot_solution()

if __name__ == '__main__':
    main()