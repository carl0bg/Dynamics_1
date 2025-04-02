import numpy as np
import math
import matplotlib.pyplot as plt


def u(x):
    return 10 + 90 * (x ** 2)


def phi(x):
    return -2110 + 450 * (x ** 2)


def solve(n):
    mu1 = 10
    mu2 = 100

    kappa1 = kappa2 = 0

    h = 1 / n

    A = 12 / (h ** 2)
    B = 12 / (h ** 2)
    C = 24 / (h ** 2) + 5

    ai = [kappa1]
    bi = [mu1]

    xi = [0]

    for i in range(1, n):
        _xi = i * h
        xi.append(_xi)
        a = B / (C - A * ai[i - 1])
        b = (phi(_xi) + A * bi[i - 1]) / (C - A * ai[i - 1])

        ai.append(a)
        bi.append(b)

    xi.append(n * h)

    yn = (kappa2 * bi[n - 1] + mu2) / (1 - kappa2 * ai[n - 1])
    yi = [yn]

    for i in range(n - 1, -1, -1):
        yi.append(ai[i] * yi[n - i - 1] + bi[i])

    yi = yi[::-1]

    return xi, yi


def solve_opt1(n, optimize_a):
    mu1 = 10
    mu2 = 100

    kappa1 = kappa2 = 0

    h = 1 / n

    hs = h ** 2 / 12

    A = 12 / (h ** 2)
    B = 12 / (h ** 2)
    C = 24 / (h ** 2) + 5

    ai = [kappa1]
    bi = [mu1]

    xi = [0]

    for i in range(1, n):
        _xi = i * h
        xi.append(_xi)
        if optimize_a:
            a = 1 / ((hs * 5 - ai[i - 1]) + 2)
        else:
            a = B / (C - A * ai[i - 1])
        b = (hs * phi(_xi) + bi[i - 1]) * a
        ai.append(a)
        bi.append(b)

    xi.append(n * h)

    yn = mu2
    yi = [yn]

    for i in range(n - 1, -1, -1):
        yi.append(ai[i] * yi[n - i - 1] + bi[i])

    yi = yi[::-1]

    return xi, yi


def make_plot(i, xs, ys, title):
    plt.subplot(2, 2, i)
    plt.title(title)

    # plt.xticks(np.arange(0, 1, 1 / n))
    plt.yticks(np.arange(0, 110, 10))

    plt.grid()
    plt.plot(np.array(xs), np.array(ys))


def z(xs, ys):
    d = []
    for i in range(len(xs)):
        d.append(math.fabs(u(xs[i]) - ys[i]))
    return max(d)


if __name__ == '__main__':
    s = input('Enter grid size: ')

    if len(s) > 0:
        n = int(s)

        uxs = np.linspace(0, 1, n)
        uys = u(uxs)

        vxs, vys = solve(n)
        print(f'z = {z(vxs, vys)}')

        vxs_opt1, vys_opt1 = solve_opt1(n, False)
        vxs_opt2, vys_opt2 = solve_opt1(n, True)
        print(f'z_opt1 = {z(vxs_opt1, vys_opt1)}, z_opt2 = {z(vxs_opt2, vys_opt2)}')

        make_plot(1, uxs, uys, 'U(x)')
        make_plot(2, vxs, vys, 'V(x)')
        make_plot(3, vxs_opt1, vys_opt1, 'V(x) (Opt 1)')
        make_plot(4, vxs_opt2, vys_opt2, 'V(x) (Opt 2)')

        plt.show()
    else:
        n = 10
        ns = []
        zs = []
        zs_opt1 = []
        zs_opt2 = []
        while n <= 10000000:
            vxs, vys = solve(n)
            vxs_opt1, vys_opt1 = solve_opt1(n, False)
            vxs_opt2, vys_opt2 = solve_opt1(n, True)
            zv = z(vxs, vys)
            zv_opt1 = z(vxs_opt1, vys_opt1)
            zv_opt2 = z(vxs_opt2, vys_opt2)
            print(f'n = {n}, z = {zv}, z_opt1 = {zv_opt1}, z_opt2 = {zv_opt2}')
            ns.append(n)
            zs.append(zv)
            zs_opt1.append(zv_opt1)
            zs_opt2.append(zv_opt2)
            n = n * 10

        plt.yscale('log')
        plt.plot(np.array(ns), np.array(zs), label='original')
        plt.plot(np.array(ns), np.array(zs_opt1), label='optimized (ver. 1)', linestyle='dashed')
        plt.plot(np.array(ns), np.array(zs_opt2), label='optimized (ver. 2)')
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()



