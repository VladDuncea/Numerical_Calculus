import numpy as np;



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
        s = np.dot(a[k, k + 1:], x_num[k + 1:])

        x_num[k] = (b[k] - s) / a[k, k]
    return x_num

def subs_asc_fast(a, b):
    """ (Optionala) Verifica daca matricea 'a' este patratica + compatibila cu vect 'b' """
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

def LU_pivotare(a):
    """ (Optionala) Verifica daca matricea 'a' este patratica"""
    assert a.shape[0] == a.shape[1], 'Matricea sistemului nu  este patratica!'

    n = a.shape[0] - 1

    # construim matricea P
    Pm = np.identity(n+1)

    # construim matricea L
    L = np.zeros([n+1,n+1])

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
            # interschimbare in P
            Pm[[p, k], :] = Pm[[k, p], :]

        """ Zero sub pozitia pivotului pe coloana. """
        for j in range(k + 1, n + 1):
            m = a[j, k] / a[k, k]
            # calcul L
            L[j,k] = m
            # calcul U
            a[j, :] -= m * a[k, :]

    """ Verifica compatibilitate again. """
    if a[n, n] == 0:
        raise AssertionError('Sist incompat sau nedet')

    # in a avem U in L avem L fara In
    L +=np.identity(n+1)

    return L,a,Pm


def rezolva_LU(L,U,B):
    x_num = subs_asc_fast(L, B)
    x_num1 = subs_desc_fast(U, x_num)
    return x_num1


A = np.array([[2.0,1.0,3.0],
              [6.0,-1.0,0.0],
              [0.0,2.0,1.0]])

b = np.array([2.0,1.0,3.0])

L = np.array([[1,0,0],
              [3,1,0],
              [0,-0.5,1]])
U= np.array([[2,1,3],
             [0,-4,-9],
             [0,0,-3.5]])

C = np.matmul(L,U)
print(C)

L1,U1 = LU_pivotare(A)