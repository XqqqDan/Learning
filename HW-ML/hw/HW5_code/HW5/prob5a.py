import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# generate one-dimensional data on a line
N = 100

a = 3
b = 4
varn = 0.1

xx = np.linspace(0, 1, N)
yy = a*xx + b + np.sqrt(varn)*np.random.randn(N)

# plot observed data and true line
plt.figure()
plt.plot(xx, yy, 'x', label='observed points')
plt.plot(xx, a*xx + b, 'k--', label='true line')

## Your code
# Fit the data (xx,yy) using a least-squares estimate and plot.
# LS
A = np.vstack([xx, np.ones(len(xx))]).T    # np.ones(len(xx))
kls, bls = np.linalg.lstsq(A, yy)[0]

# Call your "fitted" data yyLS.
yyLS = kls*xx + bls
plt.plot(xx, yyLS, 'g', label='LS')

# Now fit the data (xx,yy) using a rank-1 approximation and plot.
# LR
A = np.vstack([xx, yy])

# get rid of mean
mean = np.mean(A, axis=1)

for i in range(len(A[1])):
    A[:, i] = A[:, i] - mean
# A = np.subtract(A.T, mean).T

# caculate SVD
U, sigma, VT = la.svd(A, full_matrices=False)

# Choose the 1-rank of SVD
K = 0
u = U[:, K]
v = VT[K]

Y = np.outer(u, v) * sigma[K]

# add the mean
for i in range(len(Y[1])):
    Y[:, i] = Y[:, i] + mean

yyLR = Y[1]
xxLR = Y[0]

# Call your "fitted" data xxLR and yyLR.

plt.plot(xxLR, yyLR, 'r', label='LR')

plt.legend()
plt.show()
