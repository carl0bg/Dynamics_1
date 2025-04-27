import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Размер области
Lx, Ly = 2.0, 1.0
Nx, Ny = 100, 50

# Размер ячейки
hx = Lx / (Nx - 1)
hy = Ly / (Ny - 1)

# Сетка
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Координаты выреза
cut_x_min = 0.0
cut_x_max = 1.0
cut_y_min = 0.0
cut_y_max = 0.25

def inside_cut(x, y):
    return (cut_x_min <= x <= cut_x_max) and (cut_y_min <= y <= cut_y_max)

# Всего точек
N = Nx * Ny

# Инициализация матрицы системы
A = lil_matrix((N, N))
b = np.zeros(N)

def idx(i, j):
    return j * Nx + i

# Формирование системы
for j in range(Ny):
    for i in range(Nx):
        k = idx(i, j)
        xi, yj = x[i], y[j]

        # Точка внутри выреза
        if inside_cut(xi, yj):
            A[k, k] = 1.0
            b[k] = 0.0
            continue

        # Граничные условия
        if j == 0:
            A[k, k] = 1.0
            b[k] = 0.0  # Нижняя стенка psi = 0
        elif j == Ny - 1:
            A[k, k] = 1.0
            b[k] = 1.0  # Верхняя стенка psi = 1
        elif i == 0 or i == Nx - 1:
            A[k, k] = 1.0
            b[k] = yj  # Левые и правые стенки: фиксируем psi = y
        else:
            # Внутренняя точка: дискретизация Лапласа
            A[k, idx(i, j)] = -2.0 * (1/hx**2 + 1/hy**2)
            A[k, idx(i+1, j)] = 1.0 / hx**2
            A[k, idx(i-1, j)] = 1.0 / hx**2
            A[k, idx(i, j+1)] = 1.0 / hy**2
            A[k, idx(i, j-1)] = 1.0 / hy**2

# Решение системы
A = A.tocsr()
psi = spsolve(A, b)
PSI = psi.reshape((Ny, Nx))

# Маска для выреза
mask = np.array([[inside_cut(xi, yj) for xi in x] for yj in y])
PSI = np.ma.array(PSI, mask=mask)

# График
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