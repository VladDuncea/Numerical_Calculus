import numpy as np

A = np.array([[3,0,0],
     [0,4,1.41],
     [0,1.41,5]])

B = np.array([[1,0,0],
             [0,0.81,0.57],
            [0,-0.57,0.81]])

X = np.matmul(np.matmul(B.T,A),B)

print(X)

print(np.cos(np.arctan(2*np.sqrt(2))/2))