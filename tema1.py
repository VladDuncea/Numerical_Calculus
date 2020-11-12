# DUNCEA VLAD ALEXANDRU GRUPA 344

import numpy as np
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------------
# Functii ajutatoare colectate din lab anterioare
# ---------------------------------------------------------------------------------------------------

"""Metoda bisectiei"""
def bis_meth(f, a, b, eps):
    """ Validari """
    assert a < b, 'Parametrii a si b nu respecta a<b'

    if abs(f(a)) < eps:
        return a

    x = (b - a) / 2.0 + a

    # lucky
    if f(a) == 0:
        return a
    if f(b) == 0:
        return b

    # test left side
    if f(x) * f(a) < 0:
        return bis_meth(f, a, x, eps)

    return bis_meth(f, x, b, eps)


# ---------------------------------------------------------------------------------------------------
# EX1
# ---------------------------------------------------------------------------------------------------
""" Definim functia ajutatoare pentru a gasi solutia lui sqrt(11) sqrt(x)-sqrt(11) = 0 
        echivalent cu: x^2 - 11 = 0 """
def fex1(x):
    return x**2 - 11


""" Apelam metoda bisectiei pe  functia noastra, stim ca sqrt(11) este intre sqrt(9) si sqrt(16) deci
        cautam intre 3 si 4 """

# 7 zecimale inseamna ca vrem acuratete pana la 10^-8
x_sol = bis_meth(fex1, 3., 4., 10**(-8))
# afisare
print("Aproximare radical 11 cu 7 zecimale:" + "{:.7f}".format(x_sol))

# ---------------------------------------------------------------------------------------------------
# EX2
# ---------------------------------------------------------------------------------------------------
""" Functii pentru ex 2 """
def f1ex2(x):
    return np.e ** (x - 2)

def f2ex2(x):
    return np.cos(f1ex2(x)) + 1

def f3ex2(x):
    return f1ex2(x) - f2ex2(x)

""" functia 1(e^(x-2)) este o functie exponentiala si  va avea valori de  la 0 la inf 
    functia 2(cos(e^(x-2) + 1) din cauza cosinusului poate avea valori intre 0 si 2 
        si va avea valori repetitive in acel interval deci vom cauta in functie de functia 1
    Pentru x=0 f1 este 1/e^2 
               f2 este >1 deci functia exponentiala inca nu a trecut peste f1
    Pentru x=3 f1 este e > 2 => intersectia trebuie sa se afle in acest interval
    Vom cauta in intervalul (0,3)
    Vom afisa functiile pe [-1,4] pentru a vedea frumos punctul de intersectie """


# generare valori pentru afisarea graficului
x = np.linspace(-1, 4, 1000)
# valori pentru functia exp
y1 = f1ex2(x)
# valori pentru functia cu cos
y2 = f2ex2(x)

# calcul solutie
x_sol = bis_meth(f3ex2, -1, 4, 10**(-5))

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
plt.scatter(x_sol, f1ex2(x_sol), c='black')
plt.legend(['e^(x-2)', 'cos(e^(x-2)) + 1'])
# afisare grafic
plt.show()


# ---------------------------------------------------------------------------------------------------
# EX3
# ---------------------------------------------------------------------------------------------------
""" Functie care sa rezolve prin metoda pozitiei false """
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

""" Functia """
# functia primita pt ex3
def fex3(z):
    return z ** 3 - 9 * (z ** 2) + 26 * z - 24

""" Valori """
# valori pentru afisare grafic
x = np.linspace(-5, 5, 1000)
# daca vrei sa vezi functia mai frumos pe 1,5 sunt vizibile intersectiile cu Ox
# x = np.linspace(1, 5, 1000)
y = fex3(x)

""" Motivatie intervale:
    Capete:  f(-5) = -504, f(5) = 6 => minim o sol pt ca avem fct continua
    Derivata: f'(x) = 3x^2 - 18x + 26 calculam punctele in care se f' = 0
        Avem x1 = 3 - sqrt(3)/3 si x2 = 3+ sqrt(3)/3
            approx x1 = 2.5 si x2 = 3.5
        Stim ca in aceste puncte functia isi schimba monotonia, verificam valorile in aceste puncte
            pentru a vedea daca avem schimbare de semn(am facut tabelul pe foaie)
        f(x1) ~= 0.3 ,  f(x2) ~= -0.3
    Rezulta (pe valorile aproximative): 
        intre -5 si x1 f este strict crescatoare si are val intre -504 si 0.3 => solutie
        intre x1 si x2 f este strict descrescatoare si are val intre 0.3 si -0.3 => solutie 
        intre x2 si 5 f este strict crescatoare si are val intre -0.3 si 6 => solutie
    Cautam solutiile cu met poz false in intervalele gasite"""

# interval 1: (-5,2.5), 2: (2.5,3.5), 3: (3.5,5)
eps = 10 ** (-5)
sol1, N1 = pozitie_falsa(fex3, -5, 2.5, eps)
sol2, N2 = pozitie_falsa(fex3, 2.5, 3.5, eps)
sol3, N3 = pozitie_falsa(fex3, 3.5, 5, eps)

""" Grafic """
# desenare grafic
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

# afisare puncte pe grafic
plt.scatter(sol1, fex3(sol1), c='b')
plt.scatter(sol2, fex3(sol1), c='g')
plt.scatter(sol3, fex3(sol1), c='orange')

# afisare grafic
plt.show()


# ---------------------------------------------------------------------------------------------------
# EX4
# ---------------------------------------------------------------------------------------------------
""" Functie care rezolva prin metoda secantei """
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


""" Functia """
def fex4(z):
    return z ** 3 + 2 * (z ** 2) - z - 2

""" Valori """
x = np.linspace(-3, 3, 1000)
y = fex4(x)

""" Motivatie intervale:
    Capete:  f(-3) = -8, f(3) = 40 => minim o sol pt ca avem fct continua
    Derivata: f'(x) = 3x^2 + 4x - 1 calculam punctele in care se f' = 0
        Avem x1 = -2/3 - sqrt(6)/3 si x2 = -2/3 + sqrt(6)/3
            approx x1 = -1.5 si x2 = 0.5
        Stim ca in aceste puncte functia isi schimba monotonia, verificam valorile in aceste puncte
            pentru a vedea daca avem schimbare de semn(am facut tabelul pe foaie)
        f(x1) ~= 0.625 ,  f(x2) ~= -1.875
    Rezulta (pe valorile aproximative): 
        intre -3 si x1 f este strict crescatoare si are val intre -8 si 0.625 => solutie
        intre x1 si x2 f este strict descrescatoare si are val intre 0.625 si -1.875 => solutie 
        intre x2 si 3 f este strict crescatoare si are val intre -1.875 si 6 => solutie
    Cautam solutiile cu met secantei in intervalele gasite
    Punctele de start x1 si x2 le vom da cu 0.1 mai in interiorul intervalelor pentru
        a elimina posibilele probleme aparute din aproximari(sa avem intre x0, x1  f' = 0 )"""

# calcul 3 puncte pt fex4(x) = 0
# interval 1: (-3,-1.5), 2: (-1.5,0.5), 3: (0.5,3)
sol1, N1 = secanta(fex4, -3, -1.5, -2.9, -1.6, eps)
sol2, N2 = secanta(fex4, -1.5, 0.5, -1.4, 0.4, eps)
sol3, N3 = secanta(fex4, 0.5, 3, 0.6, 2.9, eps)


# Afisare grafica a functiei
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
# afisare puncte pe grafic
plt.scatter(sol1, 0, c='b')
plt.scatter(sol2, 0, c='g')
plt.scatter(sol3, 0, c='orange')

# afisare grafic
plt.show()
