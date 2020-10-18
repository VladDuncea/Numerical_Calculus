import numpy as np
import matplotlib.pyplot as plt


# define function
def f1(x):
    return (x ** 3) - 7 * (x ** 2) + 14 * x - 6


# get values for plot
y = f1(np.arange(0.0, 4.0, 0.1))
x = np.arange(0.0, 4.0, 0.1)

# plot initial graph (a)
plt.plot(x, y, lw=2)
plt.title("Metoda Bisectiei")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()


# find numeric solution

def bis_meth(a, b, eps):
    if (b - a) < eps:
        return b

    x = (b - a) / 2.0 + a

    # lucky
    if f1(a) == 0:
        return a
    if f1(b) == 0:
        return b

    # test left side
    if f1(x) * f1(a) < 0:
        return bis_meth(a, x, eps)

    return bis_meth(x, b, eps)

#b1
b1 = bis_meth(0, 1, 10 ** (-5))
print(b1)
#b2
b2 = bis_meth(1, 3.2, 10 ** (-5))
print(b2)
#b3
b3 = bis_meth(3.2, 4, 10 ** (-5))
print(b3)

# c
plt.plot(x, y, lw=2)
plt.scatter(b1, f1(b1))
plt.scatter(b2, f1(b2))
plt.scatter(b3, f1(b3))
plt.title("Metoda Bisectiei")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()