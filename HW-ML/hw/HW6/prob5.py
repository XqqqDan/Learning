import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from scipy.sparse.linalg import svds
from MDS import MDS
from PCA import PCA

## Load data: 
data = loadmat("mnistSubset.mat")
X = data['X']
trueLabels = data['trueLabels'][:,0]

# grab the desired digits
X1 = X[:, trueLabels==1]
X2 = X[:, trueLabels==2]
Y = np.hstack((X1,X2))

# create distance matrix and embed using MDS
# create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# embed using MDS
Xest =  MDS(Y, 3)
ax.scatter(Xest[:, 0], Xest[:, 1], Xest[:, 2], label='MDS')

plt.legend()
plt.show()


# embed using PCA
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

Xest =  MDS(X1, 3)
ax.scatter(Xest[:, 0], Xest[:, 1], Xest[:, 2], label='MDS:digits1')
Xest =  MDS(X2, 3)
ax.scatter(Xest[:, 0], Xest[:, 1], Xest[:, 2], label='MDS:digits2')

plt.legend()

ax = fig.add_subplot(122, projection='3d')

Xest =  PCA(X1, 3)
ax.scatter(Xest[:, 0], Xest[:, 1], Xest[:, 2], label='PCA:digits1')
Xest =  PCA(X2, 3)
ax.scatter(Xest[:, 0], Xest[:, 1], Xest[:, 2], label='PCA:digits2')

plt.legend()
plt.show()