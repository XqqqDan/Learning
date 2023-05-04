import numpy as np
from scipy import linalg
import scipy.io as sio
from sklearn.preprocessing import normalize
from numpy import matlib


def compute_normals(I, L):
    M = []
    for i in range(I.shape[2]):
        im = I[:, :, i]
        if M == []:
            height, width = im.shape
            M = im.reshape(-1, 1)
        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
    M = np.asarray(M)

    P = linalg.lstsq(L.T, M.T)[0].T
    N = normalize(P, axis=1)

    N = np.reshape(N, (height, width, 3))
    print(N.shape)

    return N


if __name__ == "__main__":
    catData = sio.loadmat("cat.mat")
    I = catData["I"]
    L = catData["L"]
    M = catData["M"]
    N = compute_normals(I, L)
