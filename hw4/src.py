import numpy as np
from scipy.integrate import quad

_lambda = 1

p = lambda x: 1

q = lambda x: _lambda

f = lambda x: -2 * _lambda * np.sin(np.sqrt(_lambda) * x)

y_real = lambda x: np.sin(np.sqrt(_lambda) * x)


def get_basis_fun(i, grid, N):
    h = grid[i] - grid[i-1]

    if i == 0:
        phi = lambda x: 0 if x >= grid[1] else (grid[1] - x) / h    
    elif i == N:
        phi = lambda x: \
        0 if x <= grid[N-1] \
        else \
        (x - grid[N-1]) / h
    else:
        phi = lambda x: \
        0 if x <= grid[i-1] or x >= grid[i+1] \
        else \
        (grid[i + 1] - x) / h if x <= grid[i + 1] and x >= grid[i] \
        else \
        (x - grid[i - 1]) / h
    return phi


def get_basis_coefs(x, N, h):
    A = np.zeros((N-1, N-1))
    rhs = np.zeros(N-1)

    b_lst = []
    for j in range(1, N):
        b_lst.append(quad(lambda z: (-p(z) + q(z) * (z - x[j-1]) * (x[j] - z)), x[j-1], x[j])[0] / (h ** 2))

        A[j -1, j - 1] = (
            quad(
                lambda z: p(z) + q(z) * (z - x[j-1])**2, 
                x[j-1], 
                x[j])[0] + 
            quad(
                lambda z: p(z) + q(z) * (x[j+1] - z)**2, 
                x[j], 
                x[j+1])[0]
            ) / (h ** 2)

        rhs[j - 1] = (
            quad(lambda z: (z - x[j-1]) * (-f(z)), x[j-1], x[j])[0] 
            + quad(lambda z: (x[j+1] - z) * (-f(z)), x[j], x[j+1])[0]
            ) / h

    for j in range(1, N):
        if j > 1:
            A[j- 1, j- 2] = b_lst[j - 1]
        if j < N-1:
            A[j - 1, j] = b_lst[j]
    return np.linalg.solve(A, rhs)


for N in [10, 100, 500, 1000, 5000, 10000]:
    a, b = 0, 2*np.pi
    grid = np.linspace(a, b, N+1)
    h = (b - a) / N

    y_coeffs = get_basis_coefs(grid, N, h)

    y = lambda t: sum(y_coeffs[j-1] * get_basis_fun(j, grid, N)(t) for j in range(1, N))

    diff = np.sqrt(quad(lambda x: (y(x) - y_real(x))**2, a, b)[0])
    print(N, diff, h**2, h**2 / diff)
