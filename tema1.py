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


# -------------------------------
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


#  ----------------------------------
# EX3

# functia primita pt ex3
def fex3(z):
    return z ** 3 - 9 * (z ** 2) + 26 * z - 24


x = np.linspace(-5, 5, 1000)
# daca vrei sa vezi functia mai bine pe 1,5 sunt vizibile mai bine intersectiile cu Ox
# x = np.linspace(1, 5, 1000)
y = fex3(x)

# Afisare grafica a functiei pentru a decide care sunt 3 intervale bune si  pt ca se cere la b
plt.figure(1)
# afisare functie
plt.plot(x, y, lw=2, c='r')
# titlul graficului
plt.title("Grafic EX3")
# afisare Ox + Oy
plt.axvline(0, c='black')
plt.axhline(0, c='black')
# afisare grid
plt.grid(True)


# functie care sa rezolve prin metoda pozitiei false
def pozitie_falsa(f, a, b, eps):
    n = 1
    x0 = (a * f(b) - b * f(a)) / (f(b) - f(a))
    while True:
        # incrementam nr pasi
        n += 1
        # verificam norocul de a da de solutie
        if f(x0) == 0:
            return x0, n

        # micsoram intervalul
        if f(a) * f(b) < 0:
            b = x0
        else:
            a = x0

        # calculam noua aproximare
        x1 = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if abs(x1 - x0) / abs(x0) < eps:
            break
        # actualizam x0
        x0 = x1

    return x0, n


# calcul 3 puncte pt fex3(x) = 0
# interval 1: (-5,2.5), 2: (2.5,3.5), 3: (3.5,5)
# intervalele le-am luat prin observarea graficului
eps = 10 ** (-5)
sol1, N1 = pozitie_falsa(fex3, -5, 2.5, eps)
sol2, N2 = pozitie_falsa(fex3, 2.5, 3.5, eps)
sol3, N3 = pozitie_falsa(fex3, 3.5, 5, eps)

# afisare puncte pe grafic
plt.scatter(sol1, 0, c='b')
plt.scatter(sol2, 0, c='g')
plt.scatter(sol3, 0, c='orange')

# afisare grafic
plt.show()


# ----------------------------------------------------
# EX4
def fex4(z):
    return z ** 3 + 2 * (z ** 2) - z - 2


x = np.linspace(-3, 3, 1000)
y = fex4(x)

# Afisare grafica a functiei pentru a decide care sunt 3 intervale bune si  pt ca se cere la b
plt.figure(2)
# afisare functie
plt.plot(x, y, lw=2, c='r')
# titlul graficului
plt.title("Grafic EX4")
# afisare Ox + Oy
plt.axvline(0, c='black')
plt.axhline(0, c='black')
# afisare grid
plt.grid(True)


# functie care rezolva prin metoda secantei
def secanta(f, a, b, x0, x1, eps):
    n = 1
    while abs(x1 - x0) / abs(x0) > eps:
        # cresstere nr iteratii
        n = n + 1
        # calcul noua aproximare
        x2 = (x0 * f(x1) - x1 * f(x0)) / (f(x1) - f(x0))
        # verificare iesire din interval
        assert (a <= x2 <= b)
        # actualizare x0,x1
        x0 = x1
        x1 = x2

    return x1, n


# calcul 3 puncte pt fex4(x) = 0
# interval 1: (-3,-1.5), 2: (-1.5,0.5), 3: (0.5,3)
# intervalele le-am luat prin observarea graficului
sol1, N1 = secanta(fex4, -3, -1.5, -2.8, -1.6, eps)
sol2, N2 = secanta(fex4, -1.5, 0.5, -1.4, 0.3, eps)
sol3, N3 = secanta(fex4, 0.5, 3, 0.6, 2.8, eps)

# afisare puncte pe grafic
plt.scatter(sol1, 0, c='b')
plt.scatter(sol2, 0, c='g')
plt.scatter(sol3, 0, c='orange')

# afisare grafic
plt.show()
