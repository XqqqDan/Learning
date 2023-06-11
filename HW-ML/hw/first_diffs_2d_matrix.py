import numpy as np
from scipy.sparse import diags


def first_diffs_2d_matrix(m, n):
    Dm = diags([-1, 1], [0, 1], shape=(m, m)).toarray()
    Dm[m - 1, 0] = 1
    Im = np.eye(m)
    Dn = diags([-1, 1], [0, 1], shape=(n, n)).toarray()
    Dn[n - 1, 0] = 1
    In = np.eye(n)
    A1 = np.kron(Dn, Im)
    A2 = np.kron(In, Dm)
    A = np.vstack((A1, A2))
    return A


A = first_diffs_2d_matrix(550, 430)
print(A)
