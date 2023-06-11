import numpy as np
from scipy.io import loadmat
from sklearn import decomposition
import matplotlib.pyplot as plt


def PCA(Dmat, r):
    r"""
    Syntax:       Xest = MDS(Dmat, r)

    Inputs:
    -------
        Dmat is an (N x N) dissimilarity (distance) matrix, where N is the number of points 
       
        r is the dimension of the embedding space

    Outputs:
    -------
        Xest is the set of embedded points
    """
    # Plan A: change the dataset struction
    m, n = Dmat.shape           # (784, 400), 400 pictures
    data = Dmat
    # data = []

    # for i in range(n):
    #     temp = np.reshape(Dmat[:, i], (28, 28))
    #     u, sigma, vT = np.linalg.svd(temp)
    #     sigma = np.array([sigma])
        
    #     if i < 1 :
    #         data = sigma
    #         continue

    #     data = np.vstack((data,sigma))

    # print(data.shape)           # (400, 28)

    pca=decomposition.PCA(n_components=r)
    Xest = pca.fit_transform(data)
    # print(Xest.shape)           # (400, 3)
    
    return Xest

if __name__=='__main__':
    ## Load data: 
    data = loadmat("mnistSubset.mat")
    X = data['X']
    trueLabels = data['trueLabels'][:,0]

    # grab the desired digits
    X1 = X[:, trueLabels==1]
    X2 = X[:, trueLabels==2]
    Y = np.hstack((X1,X2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Xest1 = PCA(X1, 3)
    ax.scatter(Xest1[:, 0], Xest1[:, 1], Xest1[:, 2], label='X1')

    Xest2 = PCA(X2, 3)
    ax.scatter(Xest2[:, 0], Xest2[:, 1], Xest2[:, 2], label='X2')

    plt.show()