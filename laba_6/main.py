import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin, pi

def initial_triangle(x):
    if 1 <= x < 1.5:
        return 2 * (x - 1)
    elif 1.5 <= x <= 2:
        return 1 - 2 * (x - 1.5)
    return 0

def initial_sine(x):
    if 1 <= x <= 2:
        return 0.5 * (1 + sin(2 * pi * (x - 1) - pi / 2))
    return 0

def boundary_condition(t):
    return 0

def explicit_upwind(grid, velocity, dx, dt):
    for t in range(1, len(grid)):
        for j in range(1, len(grid[0])):
            grid[t][j] = grid[t - 1][j] - velocity * dt * (grid[t - 1][j] - grid[t - 1][j - 1]) / dx
    return grid

def explicit_centered(grid, velocity, dx, dt):
    for t in range(1, len(grid)):
        for j in range(1, len(grid[0]) - 1):
            grid[t][j] = grid[t - 1][j] - velocity * dt * (grid[t - 1][j + 1] - grid[t - 1][j - 1]) / (2 * dx)
    return grid

def implicit_step(t, grid, velocity, dx, dt):
    prev = grid[t - 1]
    curr = grid[t]
    inv_dt = 1 / dt

    alpha = []
    beta = []

    left_val = curr[0]
    curr[-1] = 0

    alpha.append(0)
    beta.append(left_val)

    A = -velocity / (4 * dx)
    B = velocity / (4 * dx)
    C = -inv_dt

    for j in range(1, len(curr) - 1):
        phi = prev[j] * inv_dt - velocity / (4 * dx) * (prev[j + 1] - prev[j - 1])
        denom = C - A * alpha[j - 1]
        alpha.append(B / denom)
        beta.append((A * beta[j - 1] - phi) / denom)

    curr[-1] = beta[-1] / (1 - alpha[-1])
    for j in reversed(range(len(curr) - 1)):
        curr[j] = alpha[j] * curr[j + 1] + beta[j]

def implicit_scheme(grid, velocity, dx, dt):
    for t in range(1, len(grid)):
        implicit_step(t, grid, velocity, dx, dt)
    return grid

def compute_exact(u0_func, velocity, time_step, dx, points, dt):
    return [u0_func(i * dx - time_step * velocity * dt) for i in range(points)]

def create_grid(dx, dt, nx, nt, initial_func, boundary_func):
    grid = [[initial_func(i * dx) for i in range(nx)]]
    for _ in range(nt - 1):
        grid.append([boundary_func(0)] * nx)
    return grid, dx, dt

# Simulation parameters
velocity = 2
time_window = 15
space_length = 10
num_x = space_length * 10
num_t = time_window * 20
dx = space_length / num_x
dt = 0.5 * dx  # Stability condition

initial_func = initial_sine

if __name__ == '__main__':
    g1, dx, dt = create_grid(dx, dt, num_x, num_t, initial_func, boundary_condition)
    g2, _, _ = create_grid(dx, dt, num_x, num_t, initial_func, boundary_condition)
    g3, _, _ = create_grid(dx, dt, num_x, num_t, initial_func, boundary_condition)

    cfl = abs(velocity) * dt / dx
    if cfl > 1:
        print(f'Unstable scheme: CFL = {cfl:.3f}')
        exit(1)

    g1 = explicit_upwind(g1, velocity, dx, dt)
    g2 = explicit_centered(g2, velocity, dx, dt)
    g3 = implicit_scheme(g3, velocity, dx, dt)

    x_coords = np.linspace(0, space_length, num_x)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    def animate(frame):
        ax.clear()
        ax.plot(x_coords, g1[frame], label='Upwind')
        ax.plot(x_coords, g3[frame], label='Implicit')
        ax.plot(x_coords, compute_exact(initial_func, velocity, frame, dx, num_x, dt), label='Exact')
        ax.legend()

    ani = animation.FuncAnimation(fig, animate, frames=num_t, interval=30)
    plt.show()
