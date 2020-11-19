import numpy as np
import matplotlib.pyplot as plt


# Functii deja implementate necesare: Introduceti orice functie scrisa de voi care rezolva direct un sistem liniar
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

def grid_discret(A, b, size=50):
    """
    Construieste un grid discret si evaleaza f in fiecare punct al gridului
    """

    # size ->  Numar de puncte pe fiecare axa
    x1 = np.linspace(-4, 6, size)  # Axa x1
    x2 = np.linspace(-6, 4, size)  # Axa x2
    X1, X2 = np.meshgrid(x1, x2)  # Creeaza un grid pe planul determinat de axele x1 si x2

    X3 = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.array([X1[i, j], X2[i, j]])  # x e vectorul ce contine coordonatele unui punct din gridul definit mai sus
            X3[i, j] = .5 * x @ A @ x - x @ b  # Evaluam functia in punctul x

    return X1, X2, X3


def grafic_f(A, b):
    """
    Construieste graficul functiei f
    """

    # Construieste gridul asociat functiei
    (X1, X2, X3) = grid_discret(A, b)

    # Defineste o figura 3D
    fig1 = plt.figure()
    ax = plt.axes(projection="3d")

    # Construieste graficul functiei f folosind gridul discret X1, X2, X3=f(X1,X2)
    ax.plot_surface(X1, X2, X3, rstride=1, cstride=1, cmap='winter', edgecolor='none')

    # Etichete pe axe
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')

    # Titlu
    ax.set_title('Graficul functiei f')

    # Afiseaza figura
    fig1.show()


def linii_nivel(A, b, levels=10):
    """
    Construieste liniile de nivel ale functiei f
    """

    # Construieste gridul asociat functiei
    (X1, X2, X3) = grid_discret(A, b)

    # Ploteaza liniile de nivel ale functiei f
    fig2 = plt.figure()
    plt.contour(X1, X2, X3, levels=levels)  # levels = numarul de linii de nivel

    # Etichete pe axe
    plt.xlabel('x1')
    plt.ylabel('x2')

    # Titlu
    plt.title('Liniile de nivel ale functiei f')

    # Afiseaza figura
    fig2.show()


def metoda_de_rezolvare(A, b):
    return 0


# Definire functie f prin matricea A si vectorul b
A = np.array([[3., 2.], [2., 6.]])  # Matrice pozitiv definita
b = np.array([[2.], [-8.]])

# Apelare functii grafic
grafic_f(A, b)
linii_nivel(A, b)

# Punctul de minim determinat prin rezolvarea sistemului A*x=b
# x_num = metoda_de_rezolvare(A,b)
# plt.scatter(x_num[0], x_num[1], s=50, c='black', marker='*')
# plt.show()