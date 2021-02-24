import numpy as np


# modul special pt Jacobi
def modul_jacobi(A):
    s = 0
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                s+=A[i,j]
    return np.sqrt(s)

# ===============================================================
# Metoda Jacobi de aproximare a valorilor proprii
# ===============================================================

def metoda_jacobi(A, epsilon=1e-5):
    assert A.shape[0] == A.shape[1]
    print("----------Metoda Jacobi----------")
    n = A.shape[0]
    nrPasi = 1
    while modul_jacobi(A) > epsilon:
        print("----------- Pasul "+str(nrPasi)+" ---------------")
        print("La pasul " + str(nrPasi) + " A este:")
        print(A)
        # aflam pozitia valorii maxime(poz in vector)
        maxx = -np.inf
        p = -1
        q = -1
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] > maxx:
                    p = i
                    q = j
                    maxx = A[i, j]

        print("La pasul "+str(nrPasi)+" am ales p="+str(p+1)+", q="+str(q+1))
        # verificam daca A[p,p] == A[q,q]
        theta = 0
        if A[p, p] == A[q, q]:
            print("A[p,p] == A[q,q] =>")
            theta = np.pi/4
        else:
            print("A[p,p] != A[q,q] =>")
            theta = np.arctan((2*A[p, q])/(A[q, q]-A[p, p]))/2
        print("theta= "+str(theta))

        # calculam c si s
        c = np.cos(theta)
        s = np.sin(theta)
        print("c="+str(c)+", s="+str(s))

        # construim matricea R
        R = np.identity(n)
        R[p, p] = R[q, q] = c
        R[p, q] = s
        R[q, p] = -s
        print("Matricea R:")
        print(R)

        # calculam noua matrice A
        A = np.matmul(np.matmul(R.T, A), R)

        nrPasi+=1

    print("A final:")
    print(A)
    return A


# apelare
A = np.array(  [[4, 1, 1],
                [1, 4, 1],
                [1, 1, 4]])

metoda_jacobi(A)
