import numpy as np
from scipy.sparse import diags
import torch

# import dask


def DataNormalize(x):
    x = torch.tensor(x, dtype=torch.float16)

    return x


def first_diffs_2d_matrix(m, n):
    D_m = diags([-1, 1], [0, 1], shape=(m, m), format=None, dtype="int8").toarray()
    D_m[m - 1, 0] = 1

    D_n = diags([-1, 1], [0, 1], shape=(n, n), format=None, dtype="int8").toarray()
    D_n[n - 1, 0] = 1

    I_n = np.eye(n, dtype="int8")
    I_m = np.eye(m, dtype="int8")

    D_n_block = np.split(D_n, 10)
    I_m_block = np.split(I_m, 10)
    I_n_block = np.split(I_n, 10)
    D_m_block = np.split(D_m, 10)

    for block1, block2 in zip(D_n_block, I_m_block):
        A1 = np.kron(block1, block2)
    for block1, block2 in zip(I_n_block, D_m_block):
        A2 = np.kron(block1, block2)

    # D_m = torch.tensor(D_m, dtype=torch.int8)
    # D_n = torch.tensor(D_n, dtype=torch.int8)

    # I_n = torch.eye(n, dtype=torch.int8)
    # I_m = torch.eye(m, dtype=torch.int8)

    # A1 = torch.kron(D_n, I_m)
    # A2 = torch.kron(I_n, D_m)

    # A1 = A1.detach().numpy()
    # A2 = A2.detach().numpy()

    A = np.vstack((A1, A2))

    print(A.shape)
    return A


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    A = first_diffs_2d_matrix(550, 340)  # 550, 340
    print(A)
