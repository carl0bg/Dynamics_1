import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from tabulate import tabulate
from typing import List, Tuple, Optional


def boundary_condition(i: int, j: int, h: float, k: float) -> float:
    """Граничные условия для задачи."""
    x = i * h
    y = j * k
    
    # Область внутри круга (не учитывается)
    if i > 6 and j < 2:
        return 0.0
    
    # Левая граница: u(0, y) = y
    if i == 0:
        return y
    
    # Правая граница: u(8, y) = 2(y - 1)
    if i == 8:
        return 2 * (y - 1)
    
    # Верхняя граница: u(x, 4) = 2
    if j == 4:
        return 2.0
    
    return 0.0


def exact_solution(x: float, y: float) -> float:
    """Точное решение (заглушка, так как аналитическое решение неизвестно)."""
    return 0.0


@dataclass
class GridParameters:
    """Параметры расчетной сетки."""
    width: float = 4.0
    height: float = 2.0
    n: int = 8       # Количество узлов по x
    m: int = 4       # Количество узлов по y
    precision: float = 1e-6
    max_iterations: int = 1000

    def __post_init__(self):
        """Вычисление производных параметров."""
        self.h = self.width / self.n  # Шаг по x
        self.k = self.height / self.m  # Шаг по y
        self.h_sq = self.h ** 2
        self.k_sq = self.k ** 2
        self.h_star = 1 / self.h_sq
        self.k_star = 1 / self.k_sq
        self.a_star = -2 * (self.h_star + self.k_star)


@dataclass
class SeidelSolver:
    """Решатель уравнения методом Зейделя с учетом особых узлов."""
    
    params: GridParameters
    grid: List[List[float]] = field(init=False)
    accuracy: List[float] = field(default_factory=list)
    
    # Особые узлы:
    boundary_nodes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(6, 1), (7, 2)])  # Узлы на границе круга
    
    invalid_nodes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(7, 1)])  # Узлы внутри круга

    def __post_init__(self):
        """Инициализация сетки с граничными условиями."""
        p = self.params
        self.grid = [
            [boundary_condition(i, j, p.h, p.k) 
            for i in range(p.n + 1)]
        for j in range(p.m + 1)]

    def solve(self) -> Tuple[int, float]:
        """Выполняет решение уравнения методом Зейделя."""
        p = self.params
        max_delta = p.precision + 1
        iterations = 0

        while iterations < p.max_iterations and max_delta > p.precision:
            max_delta = 0.0
            
            for j in range(1, p.m):
                for i in range(1, p.n):
                    if (i, j) in self.invalid_nodes:
                        continue  # Пропускаем узлы внутри круга

                    delta = self._update_node(i, j)
                    max_delta = max(max_delta, delta)

            iterations += 1
            self.accuracy.append(max_delta)
            print(f"Iteration {iterations:4d}, Δ = {max_delta:.3e}")

        self._print_results_table()
        return iterations, max_delta

    def _update_node(self, i: int, j: int) -> float:
        """Обновляет значение в узле (i,j) и возвращает изменение."""
        p = self.params
        old_value = self.grid[j][i]
        
        # Стандартное обновление для обычных узлов
        if (i, j) not in self.boundary_nodes:
            left = p.h_star * self.grid[j][i - 1]
            right = p.h_star * self.grid[j][i + 1]
            up = p.k_star * self.grid[j + 1][i]
            down = p.k_star * self.grid[j - 1][i]
            self.grid[j][i] = (left + right + up + down) / -p.a_star
            return abs(self.grid[j][i] - old_value)
        
        # Специальная обработка узлов на границе круга
        if i == 6:  # Правый узел на границе круга
            x = i * p.h
            y = j * p.k
            x_wall = math.sqrt(1 - y**2) + 4  # x-координата на окружности
            alpha = (x_wall - x) / p.h  # Доля шага до границы
            
            if alpha <= 0:  # Защита от деления на 0
                return 0.0
                
            a_star_corr = -2 * (p.k_star + p.h_star / alpha)
            self.grid[j][i] = (
                self.grid[j + 1][i] + self.grid[j - 1][i] + 
                2 * (self.grid[j][i - 1] / (1 + alpha) + 
                     self.grid[j][i + 1] / (alpha * (1 + alpha)))
            ) / -a_star_corr
        else:  # Верхний узел на границе круга (j == 2)
            x = i * p.h
            y = j * p.k
            y_wall = math.sqrt(1 - (x - 4)**2)  # y-координата на окружности
            beta = (y - y_wall) / p.k  # Доля шага до границы
            
            if beta <= 0:  # Защита от деления на 0
                return 0.0
                
            a_star_corr = -2 * (p.h_star + p.k_star / beta)
            self.grid[j][i] = (
                self.grid[j][i - 1] + self.grid[j][i + 1] + 
                2 * (self.grid[j + 1][i] / (1 + beta) + 
                     self.grid[j - 1][i] / (beta * (1 + beta)))
            ) / -a_star_corr
        
        return abs(self.grid[j][i] - old_value)

    def _print_results_table(self) -> None:
        """Выводит таблицу с результатами."""
        p = self.params
        table = []
        
        for j in range(p.m + 1):
            for i in range(p.n + 1):
                if (i, j) in self.invalid_nodes:
                    continue
                    
                x = i * p.h
                y = j * p.k
                numerical = self.grid[j][i]
                exact = exact_solution(x, y)
                error = abs(numerical - exact)
                
                table.append([
                    f"{x:.2f}", f"{y:.2f}",
                    f"{numerical:.6f}",
                    f"{exact:.6f}",
                    f"{error:.2e}"
                ])
        
        headers = ["x", "y", "Численное", "Точное", "Погрешность"]
        print(tabulate(table, headers=headers, tablefmt="grid"))
        
    def plot_solution(self) -> None:
        """Визуализирует решение."""
        p = self.params
        x = np.linspace(0, p.width, p.n + 1)
        y = np.linspace(0, p.height, p.m + 1)
        X, Y = np.meshgrid(x, y)
        Z = np.array(self.grid)

        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Контурный график решения
        cs = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        fig.colorbar(cs, label='Значение функции')
        
        # Настройка осей и сетки
        ax.set_xlabel('X координата')
        ax.set_ylabel('Y координата')
        ax.set_title('Решение уравнения методом Зейделя')
        ax.set_xlim(0, p.width)
        ax.set_ylim(0, p.height)
        ax.xaxis.set_major_locator(plt.MultipleLocator(p.h))
        ax.yaxis.set_major_locator(plt.MultipleLocator(p.k))
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Закрашенный черный круг (без границы)
        circle = plt.Circle((4, 0), 1, color='black', fill=True, linewidth=0)
        ax.add_patch(circle)
        
        plt.tight_layout()
        plt.show()


def main():
    """Основная функция для выполнения расчета."""
    # Параметры расчета
    params = GridParameters(
        width=4.0,
        height=2.0,
        n=8,
        m=4,
        precision=1e-6,
        max_iterations=1000
    )
    
    # Создание и запуск решателя
    solver = SeidelSolver(params)
    iterations, final_accuracy = solver.solve()
    
    # Вывод сводки
    summary = [
        ['Количество итераций', f'{iterations}/{params.max_iterations}'],
        ['Достигнутая точность', f'{final_accuracy:.3e}'],
    ]
    print(tabulate(summary, tablefmt='grid', colalign=('left', 'right')))
    
    # Визуализация
    solver.plot_solution()


if __name__ == '__main__':
    main()