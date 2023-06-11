import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# generate two-dimensional data on an affine plane
N = 200
varn = 0.5

w = np.array([3, 4])
xx = np.random.rand(N, 2)
yy = xx @ w + np.sqrt(varn)*np.random.randn(N)

# plot observed data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx[:, 0], xx[:, 1], yy, label='observed points')

## Your code
# Fit the data (xx,yy) using a least-squares estimate and plot.
# Call your "fitted" data yyLS.
A = np.zeros((3, 3))
for i in range(0, 100):
    A[0, 0] = A[0, 0] + xx[i][0]**2
    A[0, 1] = A[0, 1] + xx[i][0]*xx[i][1]
    A[0, 2] = A[0, 2] + xx[i][0]
    A[1, 0] = A[0, 1]
    A[1, 1] = A[1, 1] + xx[i][1]**2
    A[1, 2] = A[1, 2] + xx[i][1]
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]
    A[2, 2] = 100

b = np.zeros((3, 1))
for i in range(0, 100):
    b[0, 0] = b[0, 0] + xx[i][0]*yy[i]
    b[1, 0] = b[1, 0] + xx[i][0]*yy[i]
    b[2, 0] = b[2, 0] + yy[i]

# A_inv = np.linalg.inv(A)
# X = np.dot(A_inv, b)
X = np.linalg.lstsq(A, b)[0]
yyLS = X[0, 0] * xx[:, 0] + X[1, 0] * xx[:, 1] + X[2, 0]

ax.scatter(xx[:, 0], xx[:, 1], yyLS, label='LS')
# ax.plot_trisurf(xx[:, 0], xx[:, 1], yyLS)
# ax.plot_wireframe(xx[:, 0], xx[:, 1], yyLS, rstride=10, cstride=10)

# Now fit the data (xx,yy) using a rank-2 approximation and plot.
# Call the resulting rank-2 matrix of size 3 x N Xhat.
# A = np.vstack((xx.T, yy.T))
# pca = PCA(n_components=2)
# pca.fit(A)
# Xhat = pca.components

A = np.vstack((xx.T, yy.T))
mean = np.mean(A, axis=1)

for i in range(len(A[1])):
    A[:, i] = A[:, i] - mean

U, sigma, VT = la.svd(A, full_matrices=False)
print(sigma)

u = np.vstack((U[:, 0].T, U[:, 1].T)).T
v = np.vstack((VT[0], VT[1]))

Xhat = u @ np.diag([sigma[0], sigma[1]]) @ v

for i in range(len(Xhat[0])):
    Xhat[:, i] = Xhat[:, i] + mean

ax.scatter(Xhat[0, :], Xhat[1, :], Xhat[2, :], label='LR')
# ax.plot_trisurf(Xhat[0, :], Xhat[1, :], Xhat[2, :])

plt.legend()
plt.show()
