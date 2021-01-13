"""
Laborator #13
Tested with: numpy==1.19.4, sympy==1.6.2, matplotlib==3.3.3
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


""" Foloseste libraria sympy ca sa declari simbolic variabila 'x'. 
    Documentation: https://docs.sympy.org/latest/index.html
"""
# x = sp.symbols('x')
#
# """ Calculeaza derivata functiei f(x) = 5sin(2x) - 2cos(3x) + 11.5x simbolic.
#
#     Pasul 1. Atribuie expresia functieie unei variabile folosind sympy pentru functiile trigonometrice.
#              e.g. np.sin(x) -> sp.sin(x)
#     Pasul 2. Intr-o alta variabila stocheaza derivata calculata cu sympy. Se foloseste functia 'diff()'
#
#     Pasul 3. Printeaza la consola rezultatele si verifica corectitudinea.
# """
#
# expr = 5*sp.sin(2*x) - 2*sp.cos(3*x) + 11.5*x
# diff = expr.diff()
# print('f(x)=', expr)
# print("f''(x)=", diff)
#
# """ Ca sa poti evalua o expresie simbolica care returneaza float-uri expresia trebuie convertita.
#     Functia 'lambdify()' este utila in acest caz.
#     Converteste expresia functiei si a derivatei de mai sus si evalueaza-le in punctul x = 3.
#     Printeaza rezultatele la consola.
# """
# expr = sp.lambdify(x, expr)
# diff = sp.lambdify(x, diff)
#
# print('f(3)=', expr(3))

""" Scrie o metoda care sa primeasca ca input o expresie simbolica si variabila simbolica si care returneaza expresia
    si derivata acesteia convertite astfel incat sa returneze float-uri."""

def f_df_to_float(expresie, x):
    df = expresie.diff()
    expresie = sp.lambdify(x, expresie)
    df = sp.lambdify(x, df)

    return expresie, df

""" Foloseste metoda de mai sus pentru a calcula f(x) si df(x) pe domeniul [-1, 1] pentru functia
    f(x) = 5sin(2x) - 2cos(3x) + 11.5x.
    Pentru discretizarea domeniului foloseste un grid discret cu 100 de puncte egal departate.
"""
interval = [-1., 1.]
domeniu = np.linspace(interval[0],interval[1], 100)
x = sp.symbols('x')
functie = 5*sp.sin(2*x) - 2*sp.cos(3*x) + 11.5*x

f, df = f_df_to_float(functie, x)

y = f(domeniu)
dy = df(domeniu)


""" Creeaza o figura in care sa afisezi grficul functiei de mai sus si al derivatei acesteia pe domeniul [-1, 1].
    Figura va avea cel putin notate axele OX si OY, va contine legenda si titlu.
"""
plt.figure(0)
plt.plot(domeniu, y, c='k', linewidth= 3, linestyle='-', label='f(x)')
plt.plot(domeniu, dy, c='b', linewidth= 3, linestyle='--', label="f''(x)")
plt.grid(True)
plt.axvline(0, c='black', linewidth=1)
plt.axhline(0, c='black', linewidth=1)
plt.xlabel('x')
plt.ylabel('Values')
plt.legend()
plt.show()
""" Creaza o metoda care calculeaza diferentele finite progresive, regresive si centrale pentru prima derivata
    conform abordarii de la curs.
    Metoda va avea ca date de intrare: 'X' -> Grid-ul discret, 'Y' -> Valorile functiei in punctele din grid,
                                        'metoda' -> Metoda de calcul a diferentelor finite.  
"""


def diferente_finite(X, Y, metoda='progresive'):
    assert len(X) == len(Y)
    df = np.zeros(len(X))
    df[0] = df[-1] = np.nan

    if metoda.lower() == 'progresive':
        for i in range(1, len(X)-1):
            df[i] = (Y[i+1]-Y[i]) / (X[i+1] - X[i])
    # TODO: restul metodelor
    else:
        raise ValueError("Metoda aleasa nu exista!")

    return df

""" Calculeaza derivata functiei f(x) = 5sin(2x) - 2cos(3x) + 11.5x folosind diferente finite progresive, regresive si
    centrale pe domeniul [-1, 1] folosind o discretizare a domeniului cu 'N' puncte echidistante.
    Intr-o noua figura afiseaza: derivata exacta a functiei si derivatele obtinute cu diferente finite.
    Figura va avea cel putin notate axele OX si OY, va contine legenda si titlu.
"""
N = 30

x_discret = np.zeros(N+2)
x_discret[1:-1] = np.linspace(interval[0], interval[1], N)
x_discret[0] = x_discret[1] - (x_discret[2]-x_discret[1])
x_discret[-1] = x_discret[-2] + (x_discret[2]-x_discret[1])

y_discret = f(x_discret)

df_progresive = diferente_finite(x_discret, y_discret, metoda='progresive')

plt.figure(1)
plt.title("Aproximari N=" + str(N))
plt.plot(domeniu, dy, c='k', linewidth=2, label='derivata exacta')
plt.plot(x_discret[1:-1], df_progresive[1:-1], c='orange', linewidth=2,linestyle='--', label='aproximata progresive')
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
df_ex_dicret = df(x_discret)

plt.figure(2)
plt.title("Erori aproximare pentru N=" + str(N))
plt.plot(x_discret[1:-1], np.abs(df_ex_dicret[1:-1] - df_progresive[1:-1]), c='orange', linewidth=2, linestyle='--', label='progresive')
plt.grid(True)
plt.axvline(0, c='black', linewidth=1)
plt.axhline(0, c='black', linewidth=1)
plt.xlabel('x')
plt.ylabel('Error')
plt.legend()
plt.show()
