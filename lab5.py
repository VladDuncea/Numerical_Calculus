import numpy as np


# ---------------------------------------------------------------------------------------------
# Metoda substitutiei descendente, functionala dar nu rapida.
# ---------------------------------------------------------------------------------------------
def subs_desc_slow(a, b):
    """ (Optionala) Verifica daca matricea 'a' este patratica + compatibila cu vect 'b' """
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu  este patratica!'
    assert a.shape[0] == b.shape[0], 'Vectorul nu este compatibil'

    """ Initializeaza vectorul solutiei numerice. """
    n = b.shape[0] - 1
    x_num = np.zeros(shape=n + 1)
    """ Determina solutia numerica. """
    x_num[n] = b[n] / a[n, n]
    for k in range(n - 1, -1, -1):
        s = 0.
        for j in range(k + 1, n + 1, 1):
            s += a[k, j] * x_num[j]

        x_num[k] = (b[k] - s) / a[k, k]
    return x_num


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


# ---------------------------------------------------------------------------------------------
# Metoda de eliminare Gauss fara pivotare
# ---------------------------------------------------------------------------------------------
def meg_fara_pivotare(a, b):
    """ (Optionala) Verifica daca matricea 'a' este patratica + compatibila cu vect 'b' """
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu  este patratica!'
    assert a.shape[0] == b.shape[0], 'Vectorul nu este compatibil'

    a_ext = np.concatenate((a, b[:, None]), axis=1)
    n = b.shape[0] - 1

    for k in range(n):
        """Verificare compatib sistem, Aflam pozitia pivotului de pe coloana k"""
        if not a_ext[k:, k].any():
            raise AssertionError('sistem incompatibil sau sist compatibil nedet.')
        else:
            p = np.argmin(a_ext[k:, k] == 0)
            p += k

        """ Schimba linia 'k' cu 'p' daca pivotul nu se afla pe diagonala principala. """
        if k != p:
            a_ext[[p, k], :] = a_ext[[k, p], :
                               ]
        """ Zero sub pozitia pivotului pe coloana. """
        for j in range(k + 1, n + 1):
            m = a_ext[j, k] / a_ext[k, k]
            a_ext[j, :] -= m * a_ext[k, :]

    """ Verifica compatibilitate again. """
    if a_ext[n, n] == 0:
        raise AssertionError('Sist incompat sau nedet')

    """ Gaseste solutia numerica folosind metoda subs descendente. """
    x_num = subs_desc_fast(a_ext[:, :-1], a_ext[:, -1])

    return x_num


# ---------------------------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------------------------
A = np.array([[1., 2., 3.],
              [0., 1., 2.],
              [0., 0., 2.]])

x_sol = np.array([1., 1., 1.])
b_ = np.matmul(A, x_sol)

# x_numeric = subs_desc_slow(A,b_)
# x_numeric = subs_desc_fast(A,b_)
# print(x_numeric)

a_gauss = np.array([
    [2., 3, 0],
    [3, 4, 2],
    [1, 3, 1]
])

x_sol_gauss = np.array([1, 2, 3])
b_gauss = np.matmul(a_gauss, x_sol_gauss)

meg_fara_pivotare(a_gauss, b_gauss)

eps = 10 ** (-20)
a_limitare = np.array([
    [eps, 1.],
    [1., 1.]
])
b_limitare = np.array([1., 2.])
# x_sol = [1.,1.]

x_sol_limitare = meg_fara_pivotare(a_limitare, b_limitare)
d = 2
