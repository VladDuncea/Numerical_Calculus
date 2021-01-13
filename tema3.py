# DUNCEA VLAD ALEXANDRU GRUPA 344

import numpy as np
import matplotlib.pyplot as plt


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
# EX1
# ====================================================================================================

# Functii pentru afisare(din lab)
def grid_discret(A, b, size=100):
    """
    Construieste un grid discret si evaleaza f in fiecare punct al gridului
    """

    # size ->  Numar de puncte pe fiecare axa
    x1 = np.linspace(-4, 4, size)  # Axa x1
    x2 = np.linspace(-4, 4, size)  # Axa x2
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


def met_pas_desc(point, A, b):
    # Initializari
    r = b - np.dot(A, point)
    # vectori pentru afisarea punctelor
    pointX = []
    pointY = []
    # adaugam punctul initial
    pointX.append(point[0])
    pointY.append(point[1])

    # alg in sine
    while abs(np.prod(r)) > 10**(-10):
        # calcul alpha
        alph = np.divide(np.dot(r.T, r), np.dot(r.T,np.dot(A,r)))
        # calcul punct nou
        point = point + np.dot(alph,r)
        # adaugare punct la lista
        pointX.append(point[0])
        pointY.append(point[1])
        # calcul reziduu nou
        r = b - np.dot(A, point)

    return pointX, pointY


def met_grad_conjugati(point, A, b):
    # Initializari
    r = b - np.dot(A, point)
    d = r
    # vectori pentru afisarea punctelor
    pointX = []
    pointY = []
    # adaugam punctul initial
    pointX.append(point[0])
    pointY.append(point[1])

    # alg in sine
    while abs(np.prod(r)) > 10**(-10):
        # calucl alpha
        alph = np.divide(np.dot(r.T, r), np.dot(d.T,np.dot(A,d)))
        # calcul punct nou
        point = point + np.dot(alph, d)
        # adaugare punct la lista
        pointX.append(point[0])
        pointY.append(point[1])
        # calcul reziduu nou
        r_vechi = r
        r = r - alph*np.dot(A, d)
        #calcul beta(ajutor pt noua directie)
        beta = np.divide(np.dot(r.T,r),np.dot(r_vechi.T,r_vechi))
        # calculam noua directie
        d = r + np.dot(beta, d)

    return pointX, pointY


def Ex1():
    # Functia data
    # 8*(x**2) + 4*x*y - 4*x + 13*(y**2) + 7*y
    # O desfacem in A,b
    A = np.array([[16, 4.0], [4.0, 13.0]]) # A este simetrica si pozitiv definita
    b = np.array([-4, 7])

    # Afisare grafice
    grafic_f(A, b)

    """ Metoda pasului descendent """
    # calcul minim cu pas descendent
    pointX, pointY = met_pas_desc(np.array([3, 2]), A, b)
    # Construieste gridul asociat functiei
    (X1, X2, X3) = grid_discret(A, b)
    # Ploteaza liniile de nivel ale functiei f
    fig2 = plt.figure()
    plt.contour(X1, X2, X3, levels=10)  # levels = numarul de linii de nivel
    # Etichete pe axe
    plt.xlabel('x1')
    plt.ylabel('x2')
    # Titlu
    plt.title('Gasire minim cu met pasului descendent')
    plt.plot(pointX, pointY, linewidth=2, label='Drum')
    plt.scatter(pointX, pointY, marker='.',  s=20, label='Puncte alese')
    # Afiseaza figura
    plt.legend()
    fig2.show()

    """ Metoda gradientilor conjugati """
    # calcul minim cu pas descendent
    pointX, pointY = met_grad_conjugati(np.array([3, 2]), A, b)
    # Construieste gridul asociat functiei
    # Ploteaza liniile de nivel ale functiei f
    fig3 = plt.figure()
    plt.contour(X1, X2, X3, levels=10)  # levels = numarul de linii de nivel
    # Etichete pe axe
    plt.xlabel('x1')
    plt.ylabel('x2')
    # Titlu
    plt.title('Gasire minim cu met gradientilor conjugati')
    plt.plot(pointX, pointY, linewidth=2, label='Drum')
    plt.scatter(pointX, pointY, marker='.', s=20, label='Puncte alese')
    # Afiseaza figura
    plt.legend()
    fig3.show()



#Apelare ex1
# Ex1()

# ====================================================================================================
# EX2
# ====================================================================================================
# ===============================================================
# Eroarea de trunchiere
# ===============================================================
def eroare_trunchiere(y_approx, y_adev):
    return np.abs(y_approx-y_adev)

# ===============================================================
# Eroarea maxima de trunchiere
# ===============================================================
def eroare_maxima_trunchiere(y_approx, y_adev):
    return np.max(eroare_trunchiere(y_approx,y_adev))


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


# Functia care trebuie aproximata
def fex2(x):
    """ Functia din exercitiu. """
    y = -5 * np.sin(-4*x) + 5*np.cos(3*x) - 6.38*x
    return y


# ex2 in sine
def ex2():
    # Intervalul dat
    interval = [-np.pi, np.pi]  # [a, b]

    x_domain = np.linspace(interval[0], interval[1], 200)  # Discretizare domeniu (folosit pentru plotare)
    y_values = fex2(x_domain)  # Valorile functiei exacte in punctele din discretizare

    # Afisare grafic figure
    plt.figure(0)
    plt.plot(x_domain, y_values, c='k', linewidth=2, label='Functie exacta')
    plt.xlabel('x')
    plt.ylabel('y = f(x)')
    plt.grid()

    # eroarea maxima dorita
    err_dorit = 10**(-5)
    # eroarea maxima obtinuta (intial orice mai mare ca err dorita)
    err_max = err_dorit + 1

    # Gradul maxim polinomului(initial 2)
    N = 2   # Va creste pana avem eroarea dorita
    while err_max > err_dorit:
        # oprire in caz ca ajungem la un N prea mare
        if N > 100:
            raise AssertionError("Nu am reusit sa ajungem la gradul de aproximare dorit!")

        x_stiut = np.linspace(interval[0], interval[1], N + 1)  # Discretizare interval (nodurile date de client)
        y_stiut = fex2(x_stiut)  # Valorile functiei in nodurile date de client

        # Calculare discretizare polinom
        y_interp_lagrange = np.zeros(len(x_domain))  # Folosit pentru a stoca valorile aproximate
        y_interp_lagrange2 = np.zeros(len(x_domain))  # Folosit pentru a stoca valorile aproximate
        for i in range(len(x_domain)):
            y_interp_lagrange[i] = metoda_newton_polLagrange(x_stiut, y_stiut, x_domain[i])
            y_interp_lagrange2[i] = metoda_lagrange(x_stiut, y_stiut, x_domain[i])

        err_max = eroare_maxima_trunchiere(y_interp_lagrange,y_values)
        # Afisare eroare
        print("Maxim eroare interp lagrange pt N = "+str(N) + " : "+ str(err_max))
        # break
        # Crestem N
        N += 1

    """ Avem N care satisface constrangerile de aproximare """
    # Afisare date client pe grafic
    plt.scatter(x_stiut, y_stiut, marker='*', c='red', s=200, label='Date stiute')

    # Afisare grafic aprixomare
    plt.plot(x_domain, y_interp_lagrange, c='b', linewidth=1, linestyle='-', label='Metoda Lagrange calc cu Newton')
    plt.plot(x_domain, y_interp_lagrange2, c='r', linewidth=1, linestyle='-.', label='Metoda Lagrange')
    plt.title('Interpolare Lagrange, N={}'.format(N))
    plt.legend()
    plt.show()

    # Grafic eroare
    plt.figure(1)
    plt.plot(x_domain, eroare_trunchiere(y_interp_lagrange,y_values), c='k', linewidth=2, label='Eroarea')
    plt.xlabel('x')
    plt.ylabel('y = f(x)')
    plt.grid()
    plt.title('Eroarea pt Interpolare Lagrange, N={}'.format(N))
    plt.legend()
    plt.show()


# Apelare ex2
ex2()


# ====================================================================================================
# EX3
# ====================================================================================================

def spline_liniara(X, Y, pointx):
    """ Metoda de interpolare spline liniara.
    :param X: X = [X0, X1, ..., Xn]
    :param Y: [Y0=f(X0), Y1=F(X1), ..., Yn=f(Xn)]
    :param pointx: Punct in care doresti o valoare aproximata a functiei

    :return: aprox_value: Valoarea aproximata calculata interpolarea spline liniara in pointx
    """
    # PAS 1 Initializari
    n = X.shape[0] - 1
    a = np.zeros([n])
    b = np.zeros([n])

    # PAS 2 Calcul coeficienti
    for j in range(n):
        a[j] = Y[j]
        b[j] = (Y[j+1] - Y[j]) / (X[j+1] - X[j])

    # PAS 3 Gasire interval si intoarcere valoare
    for j in range(n):
        if X[j] <= pointx <= X[j+1]:

            return a[j] + b[j] * (pointx - X[j])

    return -1


def spline_patratica(X, Y):
    """ Metoda de interpolare spline patratica.
    :param X: X = [X0, X1, ..., Xn]
    :param Y: [Y0=f(X0), Y1=f(X1), ..., Yn=f(Xn)]
    :return: coeficientii aflati
    """

    n = X.shape[0] - 1

    # matricea M
    M = np.zeros([3*n, 3*n])
    # vectorul solutie
    sol = np.zeros(3*n)
    sol[0] = Y[0]
    for i in range(1, n):
        sol[2*i-1] = Y[i]
        sol[2*i] = Y[i]
    sol[2*n-1] = Y[n]

    # Construim partea 1 din M(primele 2n ecuatii)
    for i in range(n):
        M[2*i, 3*i] = X[i]*X[i]
        M[2*i, 3*i+1] = X[i]
        M[2*i, 3*i+2] = 1
        M[2 * i+1, 3 * i] = X[i+1] * X[i+1]
        M[2 * i+1, 3 * i + 1] = X[i+1]
        M[2 * i+1, 3 * i + 2] = 1

    # Construim partea 2 din M (urmatoarele n-1 ecuatii)
    for i in range(n-1):
        M[2*n+i, 3 * i] = 2*X[i+1]
        M[2*n+i, 3 * i + 1] = 1

        M[2*n+i, 3 * i + 3] = -2*X[i+1]
        M[2*n+i, 3 * i + 4] = -1

    # Adaugam ultima linie (ca sa avem 3n ecuatii)
    M[3*n - 1, 0] = 1

    coef = meg_pivot_total(M, sol)

    return coef


def rez_spline_patratica(coef, X, pointx):
    n = X.shape[0] - 1

    # cautam intre ce puncte este punctul dat
    i = 0
    for i in range(n):
        if X[i] <= pointx <= X[i+1]:
            break
    a = coef[3*i]
    b = coef[3*i+1]
    c = coef[3*i+2]
    return a*pow(pointx, 2) + b*pointx + c


def spline_cubica_curs(X, Y):
    # Dimensiunea
    n = X.shape[0] - 1
    # Distanta dintre doua puncte
    h = X[1]-X[0]

    # Calcul b (Y_b este vectorul valorilor, B este matricea sistemului)
    B = np.zeros([n+1,n+1])
    Y_b = np.zeros([n + 1])
    B[0,0] = 1
    B[n,n] = 1
    Y_b[0] = fex3deriv(X[0])
    Y_b[n] = fex3deriv(X[n])
    for i in range(1,n):
        B[i, i-1] = 1
        B[i, i] = 4
        B[i, i+1] = 1
        Y_b[i] = (3/h) * (Y[i+1] - Y[i-1])

    # Aflare coef b prin met gauss
    b = meg_pivot_total(B, Y_b)

    # Calcul c si d
    c = np.zeros([n])
    d = np.zeros([n])
    for i in range(n):
        d[i] = (-2/pow(h, 3))*(Y[i+1] - Y[i]) + pow(h, -2)*(b[i+1] + b[i])
        c[i] = (3/pow(h, 2))*(Y[i+1] - Y[i]) - (b[i+1] + 2*b[i])/h

    # asezare coeficienti in un vector de dimensiune 4n
    coef = np.zeros([4 * n])
    for i in range(n):
        coef[4 * i] = Y[i]
        coef[4 * i + 1] = b[i]
        coef[4 * i + 2] = c[i]
        coef[4 * i + 3] = d[i]

    return coef


def rez_spline_cubica(coef, X, pointx):
    n = X.shape[0] - 1

    # cautam intre ce puncte este punctul dat
    i = 0
    for i in range(n):
        if X[i] <= pointx <= X[i+1]:
            break
    a = coef[4*i]
    b = coef[4*i+1]
    c = coef[4*i+2]
    d = coef[4*i+3]

    val = pointx - X[i]

    return d*pow(val, 3) + c*pow(val, 2) + b*val + a


# Functia care trebuie aproximata
def fex3(x):
    """ Functia din exercitiu. """
    return -5 * np.sin(4*x) + 2*np.cos(-2*x) - 13.79*x


# Derivata functiei
def fex3deriv(x):
    return -20 * np.cos(4 * x) + 4*np.sin(-2 * x) - 13.79


# ex3 in sine
def ex3():
    # Intervalul dat
    interval = [-np.pi, np.pi]  # [a, b]

    x_domain = np.linspace(interval[0], interval[1], 200)  # Discretizare domeniu (folosit pentru plotare)
    y_values = fex3(x_domain)  # Valorile functiei exacte in punctele din discretizare

    # Afisare grafic figure
    plt.figure(0)
    plt.plot(x_domain, y_values, c='k', linewidth=2, label='Functie exacta')
    plt.xlabel('x')
    plt.ylabel('y = f(x)')
    plt.grid()

    # eroarea maxima dorita NU ATINGE 10^(-5)
    err_dorit = 10**(-4)
    # eroarea maxima obtinuta (intial orice mai mare ca err dorita)
    err_max = err_dorit + 1

    # Gradul maxim al polinomului(initial 3)
    N = 3   # Va creste pana avem eroarea dorita
    while err_max > err_dorit:
        # oprire in caz ca ajungem la un N prea mare
        if N > 100:
            raise AssertionError("Nu am reusit sa ajungem la gradul de aproximare dorit!")

        x_stiut = np.linspace(interval[0], interval[1], N + 1)  # Discretizare interval (nodurile date de client)
        y_stiut = fex3(x_stiut)  # Valorile functiei in nodurile date de client

        # Calculare discretizare polinom
        y_interp = np.zeros(len(x_domain))  # Folosit pentru a stoca valorile aproximate
        # calcul coeficienti doar o data(mult mai rapid)
        coef = spline_cubica_curs(x_stiut, y_stiut)
        for i in range(len(x_domain)):
            y_interp[i] = rez_spline_cubica(coef, x_stiut, x_domain[i])

        err_max = eroare_maxima_trunchiere(y_interp,y_values)
        # Afisare de testare
        print("Maxim eroare spline cubica pt N = "+str(N) + " : "+ str(err_max))

        # Crestem N
        N += 1

    """ Avem N care satisface constrangerile de aproximare """
    # Afisare date client pe grafic
    plt.scatter(x_stiut, y_stiut, marker='*', c='red', s=200, label='Date stiute')

    # Afisare grafic aprixomare
    plt.plot(x_domain, y_interp, c='b', linewidth=1, linestyle='-', label='Spline Cubica')
    plt.title('Interpolare Spline Cubica, N={}'.format(N))
    plt.legend()
    plt.show()

    # Grafic eroare
    plt.figure(1)
    plt.plot(x_domain, eroare_trunchiere(y_interp,y_values), c='k', linewidth=2, label='Eroarea')
    plt.xlabel('x')
    plt.ylabel('y = f(x)')
    plt.grid()
    plt.title('Eroarea pt Interpolare Spline Cubica, N={}'.format(N))
    plt.legend()
    plt.show()


# Apel ex3
# ex3()
