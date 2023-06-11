import numpy as np
from lsgd import lsgd

# problem parameters
m = 100
n = 50
sigma = 0.1
np.random.seed(0)

# set up A, x, and b
A = np.random.randn(m, n)
xtrue = np.random.rand(n, 1)
b = A @ xtrue + sigma*np.random.randn(m, 1)

# parameters for lsgd
mu = 1/np.linalg.norm(A, ord=2)**2
x0 = np.zeros((n, 1))
numIter = 300

# compare results
xgd = lsgd(A, b, mu, x0, numIter)
xh = np.linalg.lstsq(A, b)[0]

print('xgd - xhat: ', np.linalg.norm(xgd - xh))
print('xhat - xtrue: ', np.linalg.norm(xh - xtrue))
print('xgd - xtrue: ', np.linalg.norm(xgd - xtrue))
