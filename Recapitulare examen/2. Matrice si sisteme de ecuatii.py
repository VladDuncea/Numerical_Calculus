import numpy as np

# ===============================================================
# Metoda de eliminare Gauss cu pivotare partiala
# ===============================================================

def subs_desc_fast(matrix):
    # matrix = matrice inferior triunghiulară, extinsa
    solution = np.zeros(4)
    #calculăm valoarea elementului de pe ultima linie din matrice
    solution[3] = matrix[3, 4]/matrix[3, 3]
    for i in range(3):
        # pentru o ecuație de forma xk*akk+...+xn*akn = bk,
        # in acest punct, stim deja xi, cu k<i<=n, deci putem rescrie:
        # [bk-(xn*akn+...+ak(k-1)*x(k-1)]/akk.
        solution[-(i+2)] = (matrix[-(i+2), 4] - np.sum(matrix[-(i+2), 0:4]*solution))/matrix[-(i+2), -(i+3)]
    return solution
def subsAsc(matrix):
    # matrix = matrice inferior triunghiulară, extinsa
    solution = np.zeros(4)
    #calculăm valoarea elementului de pe prima linie din matrice
    solution[0] = matrix[0, 4]/matrix[0, 0]
    for i in range(3):
        # raționamentul pentru această formulă este asemănător cu cel pentru subDesc
        solution[i+1] = (matrix[i+1, 4] - np.sum(matrix[i+1, 0:4]*solution))/matrix[i+1, i+1]
    return solution
def meg_fara_pivot(a,b):
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu este patratica'
    assert a.shape[0] == b.shape[0], 'Matricea sistemului si vectorul b nu sunt compatibile'
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
def meg_pivot_part(a, b):
    """Verific daca matricea 'a' este patratica + compatibila cu vectorul 'b'"""
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu este patratica'
    assert a.shape[0] == b.shape[0], 'Matricea sistemului si vectorul b nu sunt compatibile'
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


