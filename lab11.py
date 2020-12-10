import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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
        up = np.prod(np.subtract(vect_x[:i], X[:i])) * np.prod(np.subtract(vect_x[i + 1:], X[i + 1:]))
        down = np.prod(np.subtract(vect_xk[:i], X[:i])) * np.prod(np.subtract(vect_xk[i + 1:], X[i + 1:]))
        L[i] = up / down

    return np.sum(np.dot(Y, L.T))


def spline_liniara(X, Y, pointx):
    """ Metoda de interpolare spline liniara.
    :param X: X = [X0, X1, ..., Xn]
    :param Y: [Y0=f(X0), Y1=F(X1), ..., Yn=f(Xn)]
    :param pointx: Punct in care doresti o valoare aproximata a functiei

    :return: aprox_value: Valoarea aproximata calculata interpolarea spline liniara in pointx
    """
    # PAS 1
    n = X.shape[0] - 1
    a =np.zeros([n])
    b = np.zeros([n])

    # PAS 2
    for j in range(n):
        a[j] = Y[j]
        b[j] = (Y[j+1] - Y[j]) / (X[j+1] - X[j])

    # PAS 3
    for j in range(n):
        if X[j] <= pointx <= X[j+1]:

            return a[j] + b[j] * (pointx - X[j])

    return -1


# ======================================================================================================================
# Datele clientului
# ======================================================================================================================
# Nodurile de interpolare
x_client_all = [55., 69, 75, 81, 88, 91, 95, 96, 102, 108, 116, 126, 145, 156, 168, 179, 193, 205,
                222, 230, 235, 240, 242, 244, 253, 259]
y_client_all = [162., 176, 188, 209, 229, 238, 244, 256, 262, 259, 254, 260, 262, 265, 263, 260, 259,
                252, 244, 239, 233, 227, 226, 224, 224, 221]


# Date de simulare
select_from = 6  # TODO: Selecteaza alte valori ('1' ca sa afiseze toate datele)
                 # Extrage date client din 'select_from' in 'select_from' (simulari)
x_client = []
y_client = []
for i in range(len(x_client_all)):
    if i % select_from == 0:
        x_client.append(x_client_all[i])
        y_client.append(y_client_all[i])

x_client = np.array(x_client)
y_client = np.array(y_client)

N = len(x_client) - 1  # Gradul polinomului Lagrange / numar de subintervale

x_domain = np.linspace(x_client[0], x_client[-1], 100)  # Discretizare domeniu (folosit pentru plotare)


# Afisare grafic figura
plt.figure(0)
plt.xlabel('x')
plt.ylabel('y = f(x)')

# Afisare date client pe grafic
plt.scatter(x_client, y_client, marker='*', c='red', s=5, label='Date client')

# Calculare discretizare polinom
y_interp_lagrange = np.zeros(len(x_domain))  # Folosit pentru a stoca valorile aproximate
y_interp_spline_liniara = np.zeros(len(x_domain))  # Folosit pentru a stoca valorile aproximate
for i in range(len(x_domain)):
    y_interp_lagrange[i] = metoda_lagrange(x_client, y_client, x_domain[i])  # TODO: Trebuie sa scrieti voi metoda
    y_interp_spline_liniara[i] = spline_liniara(x_client, y_client, x_domain[i])  # TODO: Trebuie sa scrieti voi metoda


# Afisare doggo
image = np.load('./frida_doggo.npy')
img = Image.open( 'caine.png' )
data = np.array( img, dtype='uint8' )

plt.imshow(data, extent=[0, 300, 0, 287])

# Afisare grafic aproximare
plt.plot(x_domain, y_interp_lagrange, c='w', linewidth=2, linestyle='-.', label='Metoda Lagrange')
plt.plot(x_domain, y_interp_spline_liniara, c='g', linewidth=2, linestyle='-', label='Spline Liniara')
plt.title('Interpolare, N={}'.format(N))
plt.legend()
plt.xlim([-1, 300])  # Limiteaza domeniul de afisare
plt.ylim([-1, 300])  # Limiteaza co-domeniul de afisare
plt.show()
