# DUNCEA VLAD ALEXANDRU GRUPA 344

import numpy as np


# ====================================================================================================
# Functii ajutatoare colectate din laboratoare
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
# Metoda de eliminare Gauss cu pivotare partiala
# ===============================================================
def meg_pivot_part(a, b):
    """Verific daca matricea 'a' este patratica + compatibila cu vectorul 'b'"""

    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu este patratica'
    assert a.shape[0] == b.shape[0], 'Matricea sistemului si vectorul b nu este patratica'

    a = a.astype(float)
    a_ext = np.concatenate((a, b[:, None]), axis=1)
    n = b.shape[0] - 1
    for k in range(n):
        """ Aflam pozitia pivotului de pe coloana k + compatibilitate sistem """

        if not a_ext[k:, k].any():
            raise AssertionError('Sistem incompatibil sau sistem comp nedeterminat')
        else:
            p = np.argmax(np.abs(a_ext[k:, k]))
            p += k

        """ SCHIMBA linia 'k' cu 'p' daca pivotul nu se afla pe diagonala principala"""
        if k != p:
            a_ext[[p, k], :] = a_ext[[k, p], :]

        """Zero sub pozitia pivotului pe coloana"""

        for j in range(k + 1, n + 1):
            m = a_ext[j, k] / a_ext[k, k]
            a_ext[j, :] -= m * a_ext[k, :]

    """ Verifica compatibilitatea again."""
    if a_ext[n, n] == 0:
        raise AssertionError('Sistem incompatibil sau sistem comp nedeterminat')

    """Gaseste solutia numerica folosind metoda substitutiei descendente"""

    x_num = subs_desc_fast(a_ext[:, :-1], a_ext[:, -1])

    return x_num


# ===============================================================
# Metoda de eliminare Gauss cu pivotare totala
# ===============================================================
def meg_pivot_total(a, b):
    """Verific daca matricea 'a' este patratica + compatibila cu vectorul 'b'"""
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu este patratica'
    assert a.shape[0] == b.shape[0], 'Matricea sistemului si vectorul b nu sunt compatibile'
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


# ====================================================================================================
# Exercitiul 2
# ====================================================================================================

""" Functii ajutatoare """
# Metoda de eliminare Gauss cu pivotare partiala modificata pentru a calcula determinantul
def determinant_meg_pivot_part(a):
    """Verific daca matricea 'a' este patratica"""
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu este patratica'

    a = a.astype(float)
    n = a.shape[0] - 1
    # variabila folosita pentru a determina semnul det
    contorSemn = 0

    for k in range(n):
        """Aflam pozitia pivotului de pe coloana k """

        # daca nu avem elem != 0 atunci det e 0
        if not a[k:, k].any():
            return 0
        else:
            p = np.argmax(np.abs(a[k:, k]))
            p += k

        """ SCHIMBA linia 'k' cu 'p' daca pivotul nu se afla pe diagonala principala"""
        if k != p:
            a[[p, k], :] = a[[k, p], :]
            contorSemn += 1

        """ Zero sub pozitia pivotului pe coloana """

        for j in range(k + 1, n + 1):
            m = a[j, k] / a[k, k]
            a[j, :] -= m * a[k, :]

    """ Ultima linie contine numai 0 => det 0"""
    if a[n, n] == 0:
        return 0

    """Calculeaza det prin inmultirea elem de pe diagonala"""
    val = 1
    for i in range(n + 1):
        val *= a[i, i]

    val *= (-1) ** contorSemn
    return val


""" Exercitiul in sine """
def EX1():
    """ Date de intrare"""
    A = np.array([[0, 9, 8, 6],
                  [-1, 2, 2, 3],
                  [5, 3, 9, -3],
                  [-1, -4, -2.0, 1]])
    b = np.array([112, 33, 54, -23])

    """ Verificare sistem cu solutie unica """
    if determinant_meg_pivot_part(A) == 0:
        raise AssertionError("Sistemul nu are solutie unica!")

    """ Calcul solutie """
    x_sol = meg_pivot_total(A, b)

    # verificare solutie
    if not np.array_equal(np.dot(A, x_sol), b):
        assert "Duncea a scris o mare prostie!"

    print("EX1: Solutia este:")
    print(x_sol)


""" Apel EX1 """
# EX1()

# ====================================================================================================
# Exercitiul 2
# ====================================================================================================

""" Functie ajutatoare """
# ============================================================
# Calcul inversa
# ============================================================
def calculeaza_inversa(a):
    """ Initializeaza o matrice pentru stocarea solutiei -> O sa fie inversa """
    """ Genereaza matricea de vectori din dreapta egalului cu matricea identitate """
    n = a.shape[0]
    idm = np.identity(n)
    inversa = np.zeros((n, n))

    """ Afla fiecare vector coloana din inversa folosind una dintre metodele de eliminare GAUSS  """
    for i in range(n):
        # calculy cu meg pivot total pentru a avea erorile minime
        inversa[:, i] = meg_pivot_total(a, idm[:, i])

    return inversa


# ============================================================
# Exercitiul in sine
# ============================================================
def EX2():
    """ Date de intrare """
    B = np.array([[0, 7, -3, -7],
                  [1, -1, -3, 8],
                  [-9, 1, 3, -5],
                  [6, -8, 1.0, 7]])
    """ Verificare inversabila => apelam functia de la EX1 de calcul al det """
    det = determinant_meg_pivot_part(B)
    if det == 0:
        raise AssertionError("Matricea nu e inversabila!")

    """ Calcul inversa """
    sol = calculeaza_inversa(B)

    """ Verificare calcule """
    verif = np.matmul(B, sol)
    # rotunjim valorile pentru a elimina posibilele erori de calcul
    verif = np.ma.round(verif, 5)
    if not np.array_equal(verif, np.identity(B.shape[0])):
        raise AssertionError("Eroare la calculul inversei")

    """ Afisare rezultat """
    print("EX2: Inversa matricei B este: ")
    print(B)


""" Apel Ex2 """
# EX2()


# ====================================================================================================
# Exercitiul 3
# ====================================================================================================

def LU_pivotare(a):
    """ (Optionala) Verifica daca matricea 'a' este patratica"""
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu  este patratica!'
    a = a.astype(float)

    n = a.shape[0] - 1

    # construim matricea Permutare
    Pm = np.identity(n + 1)

    # construim matricea L
    L = np.zeros([n + 1, n + 1])

    for k in range(n):
        """Verificare compatib sistem, Aflam pozitia pivotului de pe coloana k"""
        if not a[k:, k].any():
            raise AssertionError('sistem incompatibil sau sist compatibil nedet.')
        else:
            p = np.argmax(np.abs(a[k:, k]))
            p += k

        """ Schimba linia 'k' cu 'p' daca pivotul nu se afla pe diagonala principala. """
        if k != p:
            # interschimbare in L
            L[[p, k], :] = L[[k, p], :]
            # interschimbare in U
            a[[p, k], :] = a[[k, p], :]
            # interschimbare in matricea permutare
            Pm[[p, k], :] = Pm[[k, p], :]

        """ Zero sub pozitia pivotului pe coloana. """
        for j in range(k + 1, n + 1):
            m = a[j, k] / a[k, k]
            # calcul L
            L[j, k] = m
            # calcul U
            a[j, :] -= m * a[k, :]

    """ Verifica compatibilitate again. """
    if a[n, n] == 0:
        raise AssertionError('Sist incompat sau nedet')

    # in a avem U in L avem L fara In
    L += np.identity(n + 1)

    return L, a, Pm


def rezolva_LU(L, U, B):
    x_num = subs_asc_fast(L, B)
    x_num1 = subs_desc_fast(U, x_num)
    return x_num1


""" Exercitiul in sine """
def EX3():
    """ Date de intrare """
    A = np.array([[0, 2, -6, -10],
                  [-1, 0, -2, -5],
                  [-9, -1, 9, -7],
                  [-8, -9, -4, 6]])
    b = np.array([-96, -51, -36, -59])

    """ Verificare matrice inversabila(cerinta a factorizarii LU) """
    det = determinant_meg_pivot_part(A)
    if det == 0:
        raise AssertionError("Matricea nu e inversabila!")

    L, U, Pm = LU_pivotare(A)

    """ Verificare corectitudine LU """
    X1 = np.matmul(L, U).round(5)
    X2 = np.matmul(Pm, A).round(5)
    val = np.array_equal(X1, X2)
    if not val:
        raise AssertionError('Calcul LU gresit!')

    """ Permutare b pentru a corespunde dupa posibilele mutari de linii in rezolvarea LU """
    b = np.matmul(Pm, b)

    """ Gasire solutie sistem """
    x_sol = rezolva_LU(L, U, b)

    """ Verificare solutie sistem """
    # permutam la loc b pentru verificare
    b = np.matmul(Pm, b)
    val = np.array_equal(b, np.matmul(A, x_sol).round(5))
    if (not val):
        raise AssertionError('Calcul solutie prin factorizare LU gresit!')

    """ Afisare rezultat """
    print("Ex3: Solutie sistem:")
    print(x_sol)


""" Apel Ex3 """
EX3()


# ====================================================================================================
# Exercitiul 4
# ====================================================================================================

""" Metoda cholesky """
def fact_cholesky(A):
    """ Verificari de baza """
    assert A.shape[0] == A.shape[1], 'Matricea sistemului nu  este patratica!'

    """ Verificare matrice simetrica """
    # calculam diferenta dintre matricea noastra si transpusa ei si o comparam cu 0 (cu o eroare acceptabila)
    if not np.all(np.abs(A - A.T) < 10**(-5)):
        raise AssertionError('Matricea nu este simetrica')

    """ Verificare matrice pozitiv definita """
    n = A.shape[0] - 1
    # folosim criteriul sylvester ( calculam det tuturor minorilor, si al matricei
    # si verificam sa fie pozitivi)
    for i in range(n+1):
        det = determinant_meg_pivot_part(A[:i+1, :i+1])
        if det <= 0:
            raise AssertionError('Matricea nu e pozitiv definita!')

    """ Valori initiale """
    # valoarea alpha din curs
    alph = A[0,0]
    # matricea L
    L = np.zeros([n+1, n+1])
    # initializare valori L
    L[0, 0] = np.sqrt(A[0, 0])
    L[1:, 0] = A[1:, 0]/L[0, 0]

    """ Calcul (formulele din curs) """
    for k in range(1, n+1):
        alph = A[k, k] - np.sum(np.power(L[k, 0:k],2))
        L[k, k] = np.sqrt(alph)
        for i in range(k+1,n+1):
            L[i, k] = (A[i, k] - np.sum(L[i, 0:k] * L[k, 0:k]))/L[k, k]

    return L

""" Exercitiul 4 functie """
def EX4():
    A = np.array([  [16, -28, 36, -36],
                    [-28, 74, -23, 63],
                    [36, -23, 181, -75],
                    [-36, 63, -75, 131]])

    L = fact_cholesky(A)
    print("EX4: Factorizarea: ")
    print(L)

""" Apel ex4 """
# EX4()