import numpy as np
import matplotlib.pyplot as plt
from lsgd import lsgd

# problem parameters
m = 100
n = 50
sigma = 0.1
np.random.seed(0)
coff = [0.1, 1, 1.9, 2]

# set up A, x, and b
A = np.random.randn(m, n)
xtrue = np.random.rand(n,1)
b = A @ xtrue + sigma*np.random.randn(m,1)
xhat = np.linalg.lstsq(A, b)[0] # xhat
nA = np.linalg.norm(A, ord=2)   # store ||A|| to save computation time below

# parameters for lsgd
x0 = np.zeros((n, 1))
numIter = 200


## YOUR CODE

def CaculateTheDis(A, b, c, x0, numIter):
    # my code
    # set parameters
    mu = c/nA**2
    x = x0
    top = 2/nA**2
    temp = 0
    result = [0]

    for i in range(numIter):
        if mu > top:
            print('Step is out range')
            break

        x = x - mu*A.T@(A@x-b)
        temp = np.linalg.norm((x - xhat), ord=2)

        if i == 0:
            result = temp
            continue

        result = np.append(result, temp)

    xgd = x

    return result

# plot the result
plt.figure()
plt.plot(np.arange(numIter), CaculateTheDis(A, b, coff[0], x0, numIter), '--', label='$\mu = 0.1 / ||A||^{2}$')
plt.plot(np.arange(numIter), CaculateTheDis(A, b, coff[1], x0, numIter), ':', label='$\mu = 1 / ||A||^{2}$')
plt.plot(np.arange(numIter), CaculateTheDis(A, b, coff[2], x0, numIter), '.-', label='$\mu = 1.9 / ||A||^{2}$')
plt.plot(np.arange(numIter), CaculateTheDis(A, b, coff[3], x0, numIter), label='$\mu = 2 / ||A||^{2}$')
plt.legend()
plt.show()

# plot ||xhat - xgd|| vs k for various values of mu
#cRange = ??     # TODO(you): Set range of c values
#data = np.zeros((numIter, len(cRange)))

## You may want to use the below if you format your data like I did
#plt.figure()
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.plot(np.arange(numIter), data[:, 0], label='$\mu = 0.1 / ||A||^{2}$')
#plt.plot(np.arange(numIter), data[:, 1], label='$\mu = 1 / ||A||^{2}$')
#plt.plot(np.arange(numIter), data[:, 2], label='$\mu = 1.9 / ||A||^{2}$')
#plt.plot(np.arange(numIter), data[:, 3], label='$\mu = 2 / ||A||^{2}$')
#plt.legend()
#plt.show()
