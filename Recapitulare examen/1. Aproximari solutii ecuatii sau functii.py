import numpy as np
import matplotlib as plt

# EX 1
def func1(x):
    return x**3-4*x+1

def metoda_bisectiei(f, l, r, eps=10**(-5)):
    print("-----------------1. METODA BISECTIEI-----------------")
    st = l
    dr = r
    found = False
    if f(l) == 0:
        return l
    if f(r) == 0:
        return r
    err = 1
    i = 0
    while err > eps:
        x_k = (st + dr) / 2
        print("Pasul "+ str(i)+": "+str(x_k))
        if f(x_k) == 0:
            sol = x_k
            found = True
            break
        if (f(st) * f(x_k)) < 0:
            dr = x_k
        else:
            if (f(x_k) * f(dr)) < 0:
                st = x_k
        x_k_nou = (st+dr)/2
        err = abs(f(x_k_nou)-f(x_k))/abs(x_k)
        print("Eroarea: "+ str(err))
        i += 1
    if not found:
        sol = (st + dr) / 2
    print("Solutia gasita cu eroarea minima: "+ str(sol))
    return sol

i1 = [1.5, 2]
metoda_bisectiei(func1, i1[0], i1[1])

# EX 2
# DE INTREBAT PCT a
# Newton - Raphson
def f2(x):
    return x**2+2*x-1
def df2(x):
    return 2*x+2

def metoda_NR(f, df, a, b, x_0, epsilon = 1e-5):
    # Verificari existenta
    print("-----------------2. METODA NEWTON RAPHSON-----------------")
    assert a < b, 'Mesaj eroare'
    assert np.sign(f(a)) * np.sign(f(b)) < 0, 'Mesaj eroare'
    x_old = x_0
    N = 1
    # Iteratiile algoritmului
    while True:
        x_new = x_old - f(x_old) / df(x_old)
        print("Pasul "+str(N) + ":" +str(x_new))
        if np.abs(f(x_new)) < epsilon:
            break
        print("|"+str(round(f(x_new),2))+"-"+str(round(f(x_old),2))+"|"+"/|"+str(round(x_old,2))+"|")
        err = abs(f(x_new) - f(x_old)) / abs(x_old)
        print("Eroarea: " + str(err))
        x_old = x_new
        N += 1
    return x_new, N

i2 = [0.1, 2]
x_new, N = metoda_NR(f2, df2, i2[0], i2[1], i2[1])

""" Grafic """
# desenare grafic
plt.figure(1)
# afisare functie
plt.plot(x, y, lw=2, c='r')
# titlul graficului
plt.title("Grafic EX3")
# afisare Ox + Oy
plt.axvline(0, c='black')
plt.axhline(0, c='black')
# afisare grid
plt.grid(True)

# afisare puncte pe grafic
plt.scatter(sol1, fex3(sol1), c='b')
plt.scatter(sol2, fex3(sol1), c='g')
plt.scatter(sol3, fex3(sol1), c='orange')

# afisare grafic
plt.show()
