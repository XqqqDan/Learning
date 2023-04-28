import numpy as np
import matplotlib.pyplot as plt
from first_diffs_2d_matrix import first_diffs_2d_matrix

# problem size
m = 30
n = 20
[cols, rows] = np.meshgrid(np.arange(n), np.arange(m))

# disk parameters
cx = 7;
cy = 19;
r = 5;

# create matrix with disk
X = (rows - cy)**2 + (cols - cx)**2 <= r**2

plt.figure()
plt.imshow(X, cmap='gray')
plt.title('f(x,y)')
plt.savefig('./fxy_python.eps')
plt.show()

# test your first diffs matrix
A = first_diffs_2d_matrix(m,n)

x = X.flatten('F').astype(int)
DFDX = A[:m*n,:] @ x
DFDY = A[m*n:2*m*n,:] @ x

plt.figure()
plt.imshow(np.reshape(DFDX, (m,n), order='F'), cmap='gray')
plt.title('df(x,y)/dx')
plt.savefig('./dfdx_python.eps')
plt.show()

plt.figure()
plt.imshow(np.reshape(DFDY, (m,n), order='F'), cmap='gray')
plt.title('df(x,y)/dy')
plt.savefig('./dfdy_python.eps')
plt.show()
