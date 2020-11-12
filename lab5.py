import numpy as np


# ====================================================================================================
#  Metoda substitutiei descendente, functionala ,dar nu rapida
# ====================================================================================================
def subs_desc_slow(a, b):
    """Verific daca matricea 'a' este patratica + compatibila cu vectorul 'b'"""

    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu este patratica'
    assert a.shape[0] == b.shape[0], 'Matricea sistemului si vectorul b nu este patratica'

    """Initializarea vectoul solutie numerica """
    n = b.shape[0] - 1
    x_num = np.zeros(shape=n + 1)

    """Determinarea solutiei numerica"""
    x_num[n] = b[n] / a[n, n]
    for k in range(n - 1, -1, -1):
        # TODO scapa de al doilea for
        # s = 0
        # for j in range(k + 1, n + 1, 1):
        #     s += a[k, j] * x_num[j]
        s = np.dot(a[k, k + 1:], x_num[k + 1:])

        x_num[k] = (b[k] - s) / a[k, k]

    return x_num


# ====================================================================================================
#  Metoda substitutiei descendente, rapida
# ====================================================================================================
def subs_desc_fast(a, b):
    """ (Optionala) Verifica daca matricea 'a' este patratica + compatibila cu vect 'b' """
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu  este patratica!'
    assert a.shape[0] == b.shape[0], 'Vectorul nu este compatibil'

    """ Initializeaza vectorul solutiei numerice. """
    n = b.shape[0] - 1
    x_num = np.zeros(shape=n + 1)
    """ Determina solutia numerica. """
    x_num[n] = b[n] / a[n, n]
    for k in range(n - 1, -1, -1):
        # TODO: scapa de al doilea for
        # s = 0.
        # for j in range(k+1,n+1,1):
        #     s += a[k, j] * x_num[j]
        s = np.dot(a[k, k + 1:], x_num[k + 1:])

        x_num[k] = (b[k] - s) / a[k, k]
    return x_num

# ====================================================================================================
# Metoda de eliminare Gauss fara pivotare
# ====================================================================================================
def meg_fara_pivot(a, b):
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
            p = np.argmin(a_ext[k:, k] == 0)
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

    x_num = subs_desc_slow(a_ext[:, :-1], a_ext[:, -1])

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

    x_num = subs_desc_slow(a_ext[:, :-1], a_ext[:, -1])

    return x_num
# ====================================================================================================
# Date
# ====================================================================================================

A = np.array([
    [1., 2., 3.],
    [0., 1., 2.],
    [0., 0., 2.]
])

x_sol = np.array([1., 1., 1.])
b_ = np.matmul(A, x_sol)

# x_numeric = subs_desc_slow(A, b_)
# print(x_numeric)


a_gauss = np.array([
    [5., 1., -6.],
    [2., 1., -1.],
    [6., 12., 1.]
])

x_sol_gauss = np.array([1., 2., 3.])
b_gauss = np.array([7., 8., 9.])

#x_numeric2 = meg_fara_pivot(a_gauss, b_gauss)
x_numeric2 = meg__pivot_part(a_gauss, b_gauss)
#print(x_numeric2)


# ====================================================================================================
# Calcul inversa
# ====================================================================================================
def calculeaza_inversa(a):
    """ Initializeaza o matrice pentru stocarea solutiei -> O sa fie inversa """
    """ Genereaza matricea de vectori din dreapta egalului cu matricea identitate """
    n = a.shape[0]
    idm = np.identity(n)
    inversa = np.zeros((n,n))

    """ Afla fiecare vector coloana din inversa folosind una dintre metodele de eliminare GAUSS  """
    for i in range(n):
        # calcul
        inversa[:,i] = meg_fara_pivot(a, idm[:,i])

    return inversa


sol = calculeaza_inversa(a_gauss)
verif = np.matmul(a_gauss,sol)
#print(verif)

print (sol)
