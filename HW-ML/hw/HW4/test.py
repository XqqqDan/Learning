import numpy as np
import scipy.io as sio

A = np.arange(4).reshape((2, 2))

B1, B2 = np.split(A, 2, axis=1)

print(A)
print(B1)
print(B2)

C1, C2 = np.array_split(B1, 2, axis=0)
C3, C4 = np.array_split(B2, 2, axis=0)

print(C1)
print(C2)
print(C3)
print(C4)