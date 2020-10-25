# DUNCEA VLAD ALEXANDRU GRUPA 344

import numpy as np
import matplotlib.pyplot as plt

# EX1

# calcul radical
x = np.sqrt(11)
# rotunjire la 7 zecimale
x = np.round(x, 7)
# afisare
print("Aproximare radical 11 cu 7 zecimale:" + str(x))


# EX2
def f1(x):
    return np.e ** (x - 2)


def f2(x):
    return np.cos(f1(x)) + 1


# generare valori pentru afisarea graficului
x = np.linspace(-1, 3, 1000)
y1 = f1(x)
y2 = f2(x)

# calcul intersectii(aici doar una)
# calculeaza f1-f2, sign ne da semnul functiei, diff ne spune unde acesta se schimba
# argwhere ne da indexul la care schimbarea de semn se intampla
idx = np.argwhere(np.diff(np.sign(y1 - y2)))

# deschidere grafic
plt.figure(0)
# afisare functie 1
plt.plot(x, y1, lw=2, c='r')
# afisare functie 2
plt.plot(x, y2, lw=2, c='b')
# titlul graficului
plt.title("Grafic EX2")

# afisare Ox + Oy
plt.axvline(0, c='black')
plt.axhline(0, c='black')
# afisare axe scalat
plt.axis('scaled')
# afisare grid
plt.grid(True)
# afisare punct intersectie
plt.scatter(x[idx], y1[idx], c='black')
plt.legend(['e^(x-2)', 'cos(e^(x-2)) + 1'])
# afisare grafic
plt.show()


# EX3

def pozitie_falsa(f, a, b, eps):
    return 0
