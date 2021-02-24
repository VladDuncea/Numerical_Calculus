import numpy as np

# ====================================================================================================
# Functii ajutatoare
# ====================================================================================================
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
        print("La pasul", k)
        print(a_ext)
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
    print("La pasul final:")
    print(a_ext)
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

# ---------------------------------------------------------------
# EX7

# a) Metoda directa
def interpolare_directa(X, Y, pointx):
    """ Metoda directa de calculare a polinomului de interpolare Lagrange.
    :param X: X = [X0, X1, ..., Xn]
    :param Y: [Y0=f(X0), Y1=F(X1), ..., Yn=f(Xn)]
    :param pointx: Punct in care doresti o valoare aproximata a functiei
    :return: aprox_value: Valoarea aproximata calculata folosind polinomul Lagrange in pointx
    """
    print("----------Interpolare directa-------------")

    # construim o matrice vandermond pt a afla coeficientii
    matr_vander = np.vander(X, len(X), True)
    print("Pas1: Matricea Vandermond: ")
    print(matr_vander)
    print("Y: " + str(Y))
    # calculam coeficientii prin MEG
    print("Pas2: Aplicam meg pe matricea vandermond + Y pt a afla coeficientii")
    coef = meg_pivot_total(matr_vander, Y)
    print("Coeficientii: " + str(coef))

    # calculam un vector 'vandermond' pentru punctul in care vrem sa calculam
    vect_x = np.vander([pointx], len(X), True)
    print("Pas3: Vectorul 'vandermond': ")
    print(vect_x)

    # ne folosim de vectorul 'vandermond' si coeficientii aflati pentru a calcula aproximarea
    aprox_value = np.sum(np.dot(coef, vect_x.T))
    print("Pas4: Calculam aproximarea")
    print("SUM(coef * vectorul 'vandermod')")
    print("Aproximare:" + str(aprox_value))

    return aprox_value


# b) Metoda Lagrange
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



# ----------------------------------------------------
# c) Metoda Newton
def metoda_newton_polLagrange(X, Y, pointx):
    """ Metoda Newton de calcularea a polinomului de interpolare Lagrange.
        :param X: X = [X0, X1, ..., Xn]
        :param Y: [Y0=f(X0), Y1=F(X1), ..., Yn=f(Xn)]
        :param pointx: Punct in care doresti o valoare aproximata a functiei
        :return: aprox_value: Valoarea aproximata calculata folosind polinomul Lagrange in pointx
        """
    # n este nr de puncte
    n = Y.shape[0]

    # A va fi matricea sistemului de ecuatii
    A = np.zeros([n,n])
    A[:, 0] = 1

    for i in range(1,n):
        prod = 1
        for j in range(1,i+1):
            prod *= X[i]-X[j-1]
            A[i, j] = prod

    # rezolvare sistem cu met subst
    C = subs_asc_fast(A,Y)

    # Calcul aproximare
    approx = C[0]
    prod = 1
    for i in range(1,n):
        prod *= pointx - X[i-1]
        approx += C[i]*prod

    return approx


# Apelari EX7 pe o functie aleasa de mine
def fex7(x):
    return np.sin(2*x) - 2*np.cos(3*x)


# Intervalul ex7
i7 = [-np.pi, np.pi]

# Nodurile de interpolare
N = 2   # Gradul maxim al polinomului

x7 = np.linspace(i7[0], i7[1], N+1)  # Discretizare interval (nodurile date de client)
y7 = fex7(x7)  # Valorile functiei in nodurile date de client

# a)
# for i in range(len(x7)):
#     print("--- Calcul y"+str(i+1)+" ----")
#     y_approx = interpolare_directa(x7, y7, x7[i])
#     print("y"+str(i+1)+": "+str(y_approx))


# ---------------------------------------------------------------
# Ex8

# a)
def fex8a(x):
    return np.log(x)

x8a = [1, np.e, pow(np.e, 2)]     # Discretizare interval (nodurile date de client)
y8a = fex8a(x8a)  # Valorile functiei in nodurile date de client
# apelam interp directa ca sa ne afiseze polinomul
interpolare_directa(x8a, y8a, x8a[0])
