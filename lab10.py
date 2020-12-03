import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# Substitutia descendenta
# ===============================================================
def subs_desc_fast(a, b):
    """ Verifica daca matricea 'a' este patratica + compatibila cu vect 'b' """
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu  este patratica!'
    assert a.shape[0] == b.shape[0], 'Vectorul nu este compatibil'

    """ Initializeaza vectorul solutiei numerice. """
    n = b.shape[0] - 1
    x_num = np.zeros(shape=n + 1)

    """ Determina solutia numerica. """
    x_num[n] = b[n] / a[n, n]
    for k in range(n - 1, -1, -1):
        s = np.dot(a[k, k + 1:], x_num[k + 1:])
        x_num[k] = (b[k] - s) / a[k, k]

    return x_num


# ===============================================================
# Substitutia ascendenta
# ===============================================================
def subs_asc_fast(a, b):
    """ Verifica daca matricea 'a' este patratica + compatibila cu vect 'b' """
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu  este patratica!'
    assert a.shape[0] == b.shape[0], 'Vectorul nu este compatibil'

    """ Initializeaza vectorul solutiei numerice. """
    n = b.shape[0] - 1
    x_num = np.zeros(shape=n + 1)

    """ Determina solutia numerica. """
    x_num[0] = b[0] / a[0, 0]
    for k in range(1, n+1):
        s = np.dot(a[k, 0:k], x_num[0:k])
        x_num[k] = (b[k] - s) / a[k, k]

    return x_num

# ===============================================================
# Metoda de eliminare Gauss cu pivotare totala
# ===============================================================
def meg_pivot_total(a, b):
    """Verific daca matricea 'a' este patratica + compatibila cu vectorul 'b'"""
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu este patratica'
    assert a.shape[0] == b.shape[0], 'Matricea sistemului si vectorul b nu sunt compatibile'

    """ Date initiale """
    a = a.astype(float)
    a_ext = np.concatenate((a, b[:, None]), axis=1)
    n = b.shape[0] - 1
    # vector pentru a invarti solutia finala
    revert = np.arange(n + 1)

    for k in range(n):
        """Aflam pozitia pivotului + compatibilitate sistem"""
        if not a_ext[k:, k:n].any():
            raise AssertionError('Sistem incompatibil sau sistem comp nedeterminat')
        else:
            # aflam pozitia valorii maxime(poz in vector)
            poz = np.argmax(np.abs(a_ext[k:, k:-1]))
            # calculam pozitia relativ la matrice
            lin = int(poz / (n + 1 - k)) + k
            col = poz % (n + 1 - k) + k

        """ SCHIMBA linia 'k' cu 'lin' daca pivotul nu se afla pe linia potrivita """
        if k != lin:
            a_ext[[lin, k], :] = a_ext[[k, lin], :]

        """ SCHIMBA coloana 'k' cu 'col' daca pivotul nu se afla pe coloana potrivita """
        if k != col:
            a_ext[:, [col, k]] = a_ext[:, [k, col]]
            # memoram modificarea pt a indica solutia corecta
            revert[[col, k]] = revert[[k, col]]

        """Zero sub pozitia pivotului pe coloana"""
        for j in range(k + 1, n + 1):
            m = a_ext[j, k] / a_ext[k, k]
            a_ext[j, :] -= m * a_ext[k, :]

    """ Verifica compatibilitatea again."""
    if a_ext[n, n] == 0:
        raise AssertionError('Sistem incompatibil sau sistem comp nedeterminat')

    """Gaseste solutia numerica folosind metoda substitutiei descendente"""
    x_num = subs_desc_fast(a_ext[:, :-1], a_ext[:, -1])

    """ Plasam valorile pe pozitiile corecte"""
    x_sol = np.zeros(n + 1)
    for i in range(n + 1):
        x_sol[revert[i]] = x_num[i]

    return x_sol

# Implementeaza metoda directa de interpolare Lagrange
def interpolare_directa(X, Y, pointx):
    """ Metoda directa de calculare a polinomului de interpolare Lagrange.
    :param X: X = [X0, X1, ..., Xn]
    :param Y: [Y0=f(X0), Y1=F(X1), ..., Yn=f(Xn)]
    :param pointx: Punct in care doresti o valoare aproximata a functiei
    :return: aprox_value: Valoarea aproximata calculata folosind polinomul Lagrange in pointx
    """

    """
    1. Creaza o metoda care returneaza un vector care contine elementul 'x' ridicat la puteri consecutive
    pornind de la 0 si pana la n.
    """


    """
    2. Folosindu-te de metoda de mai sus, scrie elementele matricei folosite in metoda directa de aflare a polinomului 
    de interpolare Lagrange.
    """
    matr_vander = np.vander(X, len(X), True)
    """
    3. Gaseste coeficientii polinomului rezolvand sistemul rezultat (matricea de la punctul 2 si valorile Y).
    """
    coef = meg_pivot_total(matr_vander, Y)
    """
    4. Foloseste metoda de la pasul 1 pentru ca crea un vector ce contine punctul in care doresti aproximarea ridicat la 
    puteri consecutive pornind de la 0 si pana la n.
    """
    vect_x = np.vander([pointx], len(X), True)
    """
    5. Folosindu-te de vectorul de la pasul 4 si coeficientii de la pasul 3, afla valoarea aproximata i.e. P(x_aprox),
    unde P este polinomul de interpolare Lagrange rezultat din metoda directa.
    """
    aprox_value = np.sum(np.dot(coef, vect_x.T))

    return aprox_value


def metoda_lagrange(X, Y, pointx):
    """ Metoda Lagrange de calcularea a polinomului de interpolare Lagrange.
    :param X: X = [X0, X1, ..., Xn]
    :param Y: [Y0=f(X0), Y1=F(X1), ..., Yn=f(Xn)]
    :param pointx: Punct in care doresti o valoare aproximata a functiei
    :return: aprox_value: Valoarea aproximata calculata folosind polinomul Lagrange in pointx
    """
    n = len(X)
    L = np.zeros(n)
    vect_x = np.full([n], pointx)
    for i in range(n):
        vect_xk = np.full([n], X[i])
        up = np.prod(np.subtract(vect_x[:i], X[:i])) * np.prod(np.subtract(vect_x[i+1:], X[i+1:]))
        down = np.prod(np.subtract(vect_xk[:i], X[:i])) * np.prod(np.subtract(vect_xk[i+1:], X[i+1:]))
        L[i] = up/down

    return np.sum(np.dot(Y, L.T))

# ======================================================================================================================
# Date exacte
# ======================================================================================================================
# Functie cunoscuta
def aplication_function(x):
    """ Functia din exercitiu. """
    y = np.sin(2*x) - 2*np.cos(3*x)

    return y

toy_function = aplication_function  # Al doilea exemplu

# Intervalul dat
interval = [-np.pi, np.pi]  # [a, b]

x_domain = np.linspace(interval[0], interval[1], 100)  # Discretizare domeniu (folosit pentru plotare)
y_values = toy_function(x_domain)  # Valorile functiei exacte in punctele din discretizare

# Afisare grafic figure
plt.figure(0)
plt.plot(x_domain, y_values, c='k', linewidth=2, label='Functie exacta')
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.grid()

# ======================================================================================================================
# Datele clientului
# ======================================================================================================================

# Nodurile de interpolare
N = 15 # Gradul maxim al polinomului

x_client = np.linspace(interval[0], interval[1], N+1)  # Discretizare interval (nodurile date de client)
y_client = toy_function(x_client)  # Valorile functiei in nodurile date de client

# Afisare date client pe grafic
plt.scatter(x_client, y_client, marker='*', c='red', s=200, label='Date client')

# Calculare discretizare polinom
y_interp_direct = np.zeros(len(x_domain))  # Folosit pentru a stoca valorile aproximate
y_interp_lagrange = np.zeros(len(x_domain))  # Folosit pentru a stoca valorile aproximate
for i in range(len(x_domain)):
    y_interp_direct[i] = interpolare_directa(x_client, y_client, x_domain[i])
    y_interp_lagrange[i] = metoda_lagrange(x_client, y_client, x_domain[i])  # TODO: Trebuie sa scrieti voi metoda


# Afisare grafic aprixomare
plt.plot(x_domain, y_interp_direct, c='r', linewidth=1, linestyle='--', label='Metoda directa')
plt.plot(x_domain, y_interp_lagrange, c='b', linewidth=1, linestyle='-.', label='Metoda Lagrange')
plt.title('Interpolare Lagrange, N={}'.format(N))
plt.legend()
plt.show()
