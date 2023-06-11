import numpy as np
from scipy import linalg
import scipy.io as sio
from sklearn.preprocessing import normalize
from numpy import matlib


def compute_normals(I, L):
    """
    Syntax:       N = compute_normals(I, L);
                   
    Inputs:       I is an (m x n x d) matrix whose d slices contain m x n
                  double-precision images of a common scene under different
                  lighting conditions
                  I是一个(m*n*d)矩阵,其d包含不同照明条件下场景的双精度图像
                  
                  L is a (3 x d) matrix such that L(:,i) is the lighting
                  direction vector for image I(:,:,i)
                  L是一个(3*d)矩阵,其中L(:,i)是图像I(:,:,i)的照明方向矢量.方向的矢量,即图像I(:,:,i)的照明方向


    Outputs:      N is an (m x n x 3) matrix containing the unit-norm surface
                  normal vectors for each pixel in the scene
                  N是一个(m*n*3)矩阵,包含场景中每个像素的单位正态表面场景中每个像素的法向量

    """
    M=[]    # 图片一维维化
    for i in range(I.shape[2]):
        im = I[:, :, i]
        if M == []:
            height, width = im.shape
            M = im.reshape((-1, 1))
        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
    M = np.asarray(M)

    P = linalg.lstsq(L.T, M.T)[0].T
    N = normalize(P, axis=1)

    N = np.reshape(N, (height, width, 3))
    print(N.shape)

    return N

if __name__ == '__main__':
    catData = sio.loadmat('cat.mat')
    I = catData['I']
    L = catData['L']
    M = catData['M']

    N = compute_normals(I, L)