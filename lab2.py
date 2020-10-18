import numpy as np
import matplotlib.pyplot as plt


# define function
def f(x):
    return np.cos(x) - x


def df(x):
    return -np.sin(x) - 1


a = 0
b = np.pi / 2
x_0 = np.pi / 4
eps = 10 ** (-5)
x = np.linspace(a, b, 100)
y = f(x)


def metoda_NR(f, df, a, b, epsilon, x_0):
    # Verificari existenta
    assert a < b, 'Mesaj eroare'
    assert np.sign(f(a)) * np.sign(f(b)) < 0, 'Mesaj eroare'

    x_old = x_0
    N = 0
    # Iteratiile algoritmului
    while True:
        x_new = x_old - f(x_old) / df(x_old)

        if np.abs(f(x_new)) < epsilon:
            break
        x_old = x_new

    return x_new, N


x_new, N = metoda_NR(f, df, a, b, eps, x_0)
print(x_new)

# plot initial graph (a)
plt.figure(0)
plt.plot(x, y, lw=2)
plt.title("Metoda N-R")

plt.axvline(0, c='black')
plt.axhline(0, c='black')
plt.axis('scaled')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.scatter(x_new, f(x_new))
plt.legend(['f(x)', 'x_num'])
plt.show()


