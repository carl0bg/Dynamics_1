import numpy as np
import matplotlib.pyplot as plt

def solve_laplace_equation(Lx, Ly, Nx, Ny, cut_x_min, cut_x_max, cut_y_min, cut_y_max, max_iter=10000, tol=1e-6):
    """
    Решает уравнение Лапласа для функции тока psi в прямоугольной области с вырезом.
    
    Параметры:
        Lx, Ly: размеры области по x и y
        Nx, Ny: количество точек сетки по x и y
        cut_x_min, cut_x_max, cut_y_min, cut_y_max: координаты выреза
        max_iter: максимальное количество итераций
        tol: допустимая погрешность для сходимости
        
    Возвращает:
        PSI: массив значений функции тока (с маской для выреза)
        X, Y: сетки координат
    """
    # Сетка
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    hx = Lx / (Nx - 1)
    hy = Ly / (Ny - 1)

    # Инициализация psi (начальное предположение)
    psi = np.zeros((Ny, Nx))

    # Маска для выреза
    mask = np.array([[ (cut_x_min <= xi <= cut_x_max) and (cut_y_min <= yj <= cut_y_max) 
                      for xi in x] for yj in y])

    # Итерационный метод Гаусса-Зейделя
    for _ in range(max_iter):
        psi_old = psi.copy()
        
        for j in range(1, Ny-1):  # Внутренние точки по y
            for i in range(1, Nx-1):  # Внутренние точки по x
                if mask[j, i]:
                    psi[j, i] = 0.0  # Внутри выреза psi = 0
                    continue
                    
                # Дискретизация Лапласа (5-точечный шаблон)
                psi[j, i] = ((psi[j, i+1] + psi[j, i-1]) / hx**2 +
                             (psi[j+1, i] + psi[j-1, i]) / hy**2) / (2/hx**2 + 2/hy**2)
        
        # Применение граничных условий
        psi[0, :] = 0.0      # Нижняя стенка
        psi[-1, :] = 1.0      # Верхняя стенка
        psi[:, 0] = y         # Левая стенка (psi = y)
        psi[:, -1] = y        # Правая стенка (psi = y)
        
        # Проверка сходимости
        if np.max(np.abs(psi - psi_old)) < tol:
            break

    # Применение маски выреза
    PSI = np.ma.array(psi, mask=mask)
    return PSI, X, Y

def plot_results(PSI, X, Y, cut_x_min, cut_x_max, cut_y_min, cut_y_max):
    """Визуализирует результаты."""
    fig, ax = plt.subplots(figsize=(10, 5))
    levels = np.linspace(0, 1, 50)
    contour = ax.contourf(X, Y, PSI, levels=levels, cmap='coolwarm')
    cbar = plt.colorbar(contour)
    cbar.set_label('$\psi(x,y)$')

    # Границы выреза
    rect = plt.Rectangle((cut_x_min, cut_y_min), cut_x_max - cut_x_min, cut_y_max - cut_y_min,
                         color='white', zorder=10)
    ax.add_patch(rect)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Обтекание выреза линиями уровня $\psi(x,y)$')
    ax.set_aspect('equal')
    plt.show()

# Параметры задачи
Lx, Ly = 2.0, 1.0
Nx, Ny = 100, 50
cut_x_min, cut_x_max = 0.0, 1.0
cut_y_min, cut_y_max = 0.0, 0.25

# Решение и визуализация
PSI, X, Y = solve_laplace_equation(Lx, Ly, Nx, Ny, cut_x_min, cut_x_max, cut_y_min, cut_y_max)
plot_results(PSI, X, Y, cut_x_min, cut_x_max, cut_y_min, cut_y_max)