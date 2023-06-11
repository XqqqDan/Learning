import numpy as np
from scipy.sparse import diags, kron, vstack
import torch


def first_diffs_2d_matrix(m, n):
    # D_m = diags([-1, 1], [0, 1], shape=(m, m)).toarray()
    # D_m[m - 1, 0] = 1
    # I_m = np.eye(m)
    # D_n = diags([-1, 1], [0, 1], shape=(n, n)).toarray()
    # D_n[n - 1, 0] = 1
    # I_n = np.eye(n)
    # # A1 = np.kron(D_n, I_m)
    # # A2 = np.kron(I_n, D_m)

    # D_n_block = np.split(D_n, 10)
    # I_m_block = np.split(I_m, 10)
    # I_n_block = np.split(I_n, 10)
    # D_m_block = np.split(D_m, 10)

    # for block1, block2 in zip(D_n_block, I_m_block):
    #     A1 = np.kron(block1, block2)
    # for block1, block2 in zip(I_n_block, D_m_block):
    #     A2 = np.kron(block1, block2)

    D_m = diags([-1, 1], [0, 1], shape=(m, m), format="lil", dtype=np.int8)
    D_m[m - 1, 0] = 1

    D_n = diags([-1, 1], [0, 1], shape=(n, n), format="lil", dtype=np.int8)
    D_n[n - 1, 0] = 1

    I_n = diags([1], [0], shape=(n, n), format="lil", dtype=np.int8)
    I_m = diags([1], [0], shape=(m, m), format="lil", dtype=np.int8)

    A1 = kron(D_n, I_m, format="lil")
    A2 = kron(I_n, D_m, format="lil")

    A = vstack([A1, A2], format="lil")

    # A = np.vstack((A1, A2))
    return A


if __name__ == "__main__":
    A = first_diffs_2d_matrix(550, 430)  # 550, 430
    print(A)
