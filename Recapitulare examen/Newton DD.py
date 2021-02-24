import numpy as np

def metoda_newton_cu_dd(X, Y, pctx):
    # n este nr de puncte
    n = Y.shape[0]
    print("------------ Metoda Newton cu DD -----------------")

    # calculam matricea Q
    print("Calculam matricea Q")
    Q = np.zeros([n, n])
    for i in range(n):
        Q[i, 0] = Y[i]
    print("Q dupa coloana 1: ")
    print(Q)

    # calculam restul coloanelor
    for j in range(1,n):
        for i in range(j,n):
            Q[i,j] = (Q[i,j-1] - Q[i-1,j-1])/(X[i]-X[i-j])
        print("Q dupa coloana " + str(j+1)+": ")
        print(Q)

    # calculam aproximarea in punctul dat
    approx = Q[0,0]
    prod = 1
    for i in range(1,n):
        prod *= (pctx - X[i-1])
        approx += Q[i,i]*prod
    print("Aproximarea in punctul "+str(pctx)+" este: "+str(approx))

    return approx


X = np.array([0,0.4,0.7])
Y = np.array([1,3,6])

metoda_newton_cu_dd(X,Y,0.5)