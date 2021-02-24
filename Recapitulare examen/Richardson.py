import numpy as np

def formula_richardson(f,hi,x,h,n):
    # matricea ajutatoare pt calculul derivatei
    Q = np.zeros([n, n])

    # pasul 1, construim coloana 1
    print("Pasul 1: construim coloana 1 din Q")
    for i in range(n):
        Q[i, 0] = hi(x, h/pow(2, i))
    print(Q)

    # pasul 2, construim restul matricei
    print("Pasul 2: construim restul matricei Q")
    for j in range(1, n):
        for i in range(j, n):
            Q[i, j] = Q[i, j-1] + (Q[i, j-1] - Q[i-1, j-1])/(pow(2, j)-1)
        print("Q dupa constructia coloanei "+str(j+1))
        print(Q)
    return Q

