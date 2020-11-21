# DUNCEA VLAD ALEXANDRU GRUPA 344

import numpy as np
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------------
# Functii ajutatoare colectate din lab anterioare
# ---------------------------------------------------------------------------------------------------
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

def subs_asc_fast(a, b):
    """ Verifica daca matricea 'a' este patratica + compatibila cu vect 'b' """
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu  este patratica!'
    assert a.shape[0] == b.shape[0], 'Vectorul nu este compatibil'

    """ Initializeaza vectorul solutiei numerice. """
    n = b.shape[0] - 1
    x_num = np.zeros(shape=n + 1)
    """ Determina solutia numerica. """
    x_num[0] = b[0] / a[0, 0]
    for k in range(1, n):
        s = np.dot(a[k, 0:k-1], x_num[0:k-1])

        x_num[k] = (b[k] - s) / a[k, k]
    return x_num

# ====================================================================================================
# Metoda de eliminare Gauss cu pivotare partiala
# ====================================================================================================
def meg__pivot_part(a, b):
    """Verific daca matricea 'a' este patratica + compatibila cu vectorul 'b'"""

    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu este patratica'
    assert a.shape[0] == b.shape[0], 'Matricea sistemului si vectorul b nu este patratica'

    a_ext = np.concatenate((a, b[:,None]), axis=1)
    n = b.shape[0] - 1
    for k in range(n):
        """Aflam pozitia pivotului de pe coloana k + compatibilitate sistem"""

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



# ---------------------------------------------------------------------------------------------------
# Exercitiul 1
# ---------------------------------------------------------------------------------------------------

""" Functii ajutatoare """
# Metoda de eliminare Gauss cu pivotare partiala modificata pentru a calcula determinantul
def determinant_meg__pivot_part(a):
    """Verific daca matricea 'a' este patratica"""
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu este patratica'

    n = a.shape[0] - 1
    # variabila folosita pentru a determina semnul det
    contorSemn = 0;

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
    for i in range(n+1):
        val *= a[i,i]

    val *= (-1) ** contorSemn
    return val


""" Exercitiul in sine """
def EX1():

    """ Date de intrare"""
    A = np.array([[0,9,8,6],
                 [-1,2,2,3],
                 [5,3,9,-3],
                 [-1,-4,-2.0,1]])
    b = np.array([112,33,54,-23])

    """ Verificare sistem cu solutie unica """
    if(determinant_meg__pivot_part(A) == 0):
        raise AssertionError("Sistemul nu are solutie unica!")


""" Apel EX1 """
EX1()