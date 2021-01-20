# TEMA 4
# DUNCEA VLAD ALEXANDRU 344

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ====================================================================================================
# EX1
# ====================================================================================================

def f_ddf_to_float(expresie, x):
    """ Metoda care sa primeasca ca input o expresie simbolica si variabila simbolica si care returneaza expresia
        si derivata a doua a acesteia convertite astfel incat sa returneze float-uri."""
    ddf = expresie.diff().diff()
    expresie = sp.lambdify(x, expresie)
    ddf = sp.lambdify(x, ddf)

    return expresie, ddf

def diferente_finite_dd(X, Y):
    assert len(X) == len(Y)
    ddf = np.zeros(len(X))
    ddf[0] = ddf[-1] = np.nan
    # formula din curs
    for i in range(1, len(X)-1):
        ddf[i] = (Y[i+1] - 2*Y[i] + Y[i-1]) / pow(X[i+1] - X[i], 2)

    return ddf

# Cod Ex1
def ex1():
    # intervalul dat in cerinta
    interval = [-np.pi/2, np.pi]
    # discretizare domeniu cu 100 de puncte
    domeniu = np.linspace(interval[0], interval[1], 100)
    # constructie functie
    x = sp.symbols('x')
    functie = sp.cos(-0.3*x)
    # aflare functie si derivata exacta
    f, ddf = f_ddf_to_float(functie, x)
    # valori exacte pt y functie si y derivata
    y = f(domeniu)
    ddy = ddf(domeniu)

    # eroarea maxima dorita
    err_dorit = 1e-5

    # eroarea maxima obtinuta (intial orice mai mare ca err dorita)
    err_max = err_dorit + 1

    # N initial - 1
    N = 2

    while err_max > err_dorit:
        # Crestem N
        N += 1
        # oprire in caz ca ajungem la un N prea mare
        if N > 400:
            raise AssertionError("Nu am reusit sa ajungem la gradul de aproximare dorit!")
        # constructie X discret
        x_discret = np.zeros(N+2)
        x_discret[1:-1] = np.linspace(interval[0], interval[1], N)
        x_discret[0] = x_discret[1] - (x_discret[2]-x_discret[1])
        x_discret[-1] = x_discret[-2] + (x_discret[2]-x_discret[1])
        # constructie Y discret
        y_discret = f(x_discret)

        # aflare df cu metoda ceruta
        ddf_dif = diferente_finite_dd(x_discret, y_discret)

        # df exacta pe X discret
        ddf_ex_discret = ddf(x_discret)

        err_max = np.max(np.abs(ddf_ex_discret[1:-1] - ddf_dif[1:-1]))

    # grafic derivata exacta/derivata noastra
    plt.figure(0)
    plt.title("Aproximare derivata a doua, N=" + str(N))
    plt.plot(domeniu, ddy, c='k', linewidth=2, label='derivata a doua exacta')
    plt.plot(x_discret[1:-1], ddf_dif[1:-1], c='orange', linewidth=2, linestyle='--', label='aproximata')
    plt.grid(True)
    plt.axvline(0, c='black', linewidth=1)
    plt.axhline(0, c='black', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    """ Calculeaza eroarea de aproximare pentru prima derivata in fiecare punct din grid-ul discret pentru fiecare
        metoda cu diferente finite. Afiseaza intr-o noua figura erorile obtinute.
    """


    plt.figure(1)
    plt.title("Erori aproximare a doua derivata, N=" + str(N))
    plt.plot(x_discret[1:-1], np.abs(ddf_ex_discret[1:-1] - ddf_dif[1:-1]), c='orange', linewidth=2, label='eroare trunchiere')
    plt.grid(True)
    plt.axvline(0, c='black', linewidth=1)
    plt.axhline(0, c='black', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


# Apel ex1
# ex1()

# ====================================================================================================
# EX2
# ====================================================================================================
# functia data in cerinta
def fex2(x):
    sigm = 1.2
    v1 = 1/(sigm * np.sqrt(2*np.pi))
    putere = -(pow(x,2)/(2*pow(sigm,2)))
    return v1*pow(np.e,putere)


def integrare(f, X, metoda='dreptunghi'):
    # variabila in care vom calcula aproximarea
    approx = 0
    # distanta dintre doua pucnte(stim ca sunt echidistante
    h = X[1] - X[0]

    if metoda.lower() == 'dreptunghi':
        # m este dimensiunea vectorului/2
        m = int(len(X)/2)
        for i in range(m):
            approx += f(X[2*i])
        approx *= 2*h
    elif metoda.lower() == 'trapez':
        # m este dimensiunea vectorului-1
        m = int(len(X) - 1)
        approx = X[0] + X[m]
        for i in range(1, m):
            approx += 2*f(X[i])
        approx *= h/2
    elif metoda.lower() == 'simpson':
        # m este dimensiunea vectorului/2
        m = int(len(X)/2)
        approx = X[0] + X[2*m]

        sum_part = 0
        for i in range(0, m):
            sum_part += f(X[2*i+1])
        approx += 4 * sum_part

        sum_part = 0
        for i in range(1, m-1):
            sum_part += f(X[2*i])
        approx += 2 * sum_part

        approx *= h/3
    else:
        raise ValueError("Metoda aleasa nu exista!")

    return approx


def ex2():
    a = -12
    b = 12
    # calcul 'exact' al integrale
    x = sp.symbols('x')
    f = (1/(1.2 * sp.sqrt(2*sp.pi))) * sp.exp(-(pow(x, 2)/(2*pow(1.2, 2))))
    adev = sp.integrate(f, (x, a, b)).evalf()
    print("Aproximare folosind metoda interna sympy: " + str(adev))

    # am ales N impar la intamplare, 25
    N = 41
    # X discret
    x_discret = np.linspace(a, b, N)
    # apel integrare pt dreptunghi
    approx1 = integrare(fex2, x_discret)
    print("Aproximare folosind metoda de cuadratura sumata a dreptunghiului: " + str(approx1))
    # apel integrare pt trapez
    approx2 = integrare(fex2, x_discret, metoda='trapez')
    print("Aproximare folosind metoda de cuadratura sumata a trapezului: " + str(approx2))
    # apel integrare pt Simpson
    approx3 = integrare(fex2, x_discret, metoda='Simpson')
    print("Aproximare folosind metoda de cuadratura sumata Simpson: " + str(approx3))

# Apel ex2
ex2()

