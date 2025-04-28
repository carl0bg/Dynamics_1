import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Параметры области
Lx, Ly = 2.0, 1.0
Nx, Ny = 100, 50
hx, hy = Lx/(Nx-1), Ly/(Ny-1)

# Сетка
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Параметры выреза
cut_x1, cut_x2 = 0.8, 1.2
cut_y1, cut_y2 = 0.3, 0.7

def in_cut(x, y):
    return (cut_x1 <= x <= cut_x2) and (cut_y1 <= y <= cut_y2)

# Построение матрицы
N = Nx * Ny
A = lil_matrix((N, N))
b = np.zeros(N)

def index(i, j):
    return j * Nx + i

# Граничные условия и заполнение матрицы
for j in range(Ny):
    for i in range(Nx):
        k = index(i, j)
        xi, yj = x[i], y[j]
        
        if in_cut(xi, yj):
            A[k, k] = 1
            b[k] = 0
            continue
            
        if j == 0 or j == Ny-1:  # Стенки
            A[k, k] = 1
            b[k] = 0
        elif i == 0:  # Вход: параболический профиль
            A[k, k] = 1
            b[k] = 1.5 * (1 - (2*yj - 1)**2)  # Максимум 1.5 в центре
        elif i == Nx-1:  # Выход
            A[k, k] = 1/hx
            A[k, index(i-1, j)] = -1/hx
            b[k] = 0
        else:  # Внутренние точки
            A[k, index(i,j)] = -2*(1/hx**2 + 1/hy**2)
            A[k, index(i+1,j)] = 1/hx**2
            A[k, index(i-1,j)] = 1/hx**2
            A[k, index(i,j+1)] = 1/hy**2
            A[k, index(i,j-1)] = 1/hy**2

# Решение системы
A = csr_matrix(A)
psi = spsolve(A, b).reshape(Ny, Nx)

# Расчёт скорости (u = ∂ψ/∂y)
u = np.gradient(psi, y, axis=0)
v = -np.gradient(psi, x, axis=1)
speed = np.sqrt(u**2 + v**2)

# Нормализация скорости (макс. значение = 1.5)
speed = speed / np.nanmax(speed) * 1.5

# Маскировка выреза
speed_masked = np.where(
    (X >= cut_x1) & (X <= cut_x2) & (Y >= cut_y1) & (Y <= cut_y2),
    np.nan, speed
)

# Визуализация
plt.figure(figsize=(12, 5))
contour = plt.contourf(X, Y, speed_masked, levels=20, cmap='jet', vmin=0, vmax=1.5)
plt.colorbar(contour, label='Скорость (нормализованная)')

# Отображение выреза
plt.fill_between([cut_x1, cut_x2], cut_y1, cut_y2, color='white')

plt.title('Корректное распределение скорости с параболическим профилем')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()