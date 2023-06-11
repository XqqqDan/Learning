import numpy as np
import matplotlib.pyplot as plt
from mae_irls import mae_irls

np.random.seed(0)
N = 100

# generate data with outliers
a = 10
b = 5

xx = np.random.rand(N)
zz = np.zeros(N)
k = int(N*0.5)
rp = np.random.permutation(N)
outlier_subset = rp[:k]
zz[outlier_subset] = 1
yy = (1-zz)*(a*xx + b + np.random.randn(N)) + zz*(20 - 20*xx + 10*np.random.randn(N))
# yy = (1-zz)*(a*xx + b) + zz*(20 - 20*xx)

# plot observed data and true line
plt.figure()
plt.plot(xx, yy, 'x', label='observed points')
plt.plot(xx, a*xx + b, 'k--', label='true line')

# plot LS fit
A = np.array([xx, np.ones(N)]).T
params = np.linalg.pinv(A) @ yy
yyLS = params[0]*xx + params[1]
plt.plot(xx, yyLS, 'g', label='LS')

## Your code
# Fit the data (xx,yy) using a least absolute error estimate and plot.
IrlsParams = mae_irls(A, yy, xx, 20)
yyIrls = IrlsParams[0]*xx + IrlsParams[1]
plt.plot(xx, yyIrls, '--', label='Irls')

plt.legend()
plt.show()
