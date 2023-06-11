import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from scipy.sparse.linalg import lsqr
from first_diffs_2d_matrix import first_diffs_2d_matrix
from compute_normals import compute_normals

# load data
# I = images   (m x n x d)
# L = lighting (3 x d) 
# M = mask     (m x n)
catData = sio.loadmat('cat.mat')
I = catData['I']
L = catData['L']
M = catData['M']
m, n, d = I.shape   # (550, 430, 12)
print(L.shape)

# split the matrix
print(I.shape)
A1, A2 = np.split(I, 2, axis=1)
B1, B2 = np.array_split(A1, 2, axis=0)
B3, B4 = np.array_split(A2, 2, axis=0)
I1, I2, I3, I4 = B1, B2, B3, B4
print(I1.shape, I2.shape, I3.shape, I4.shape)

print(M.shape)
A1, A2 = np.split(M, 2, axis=1)
B1, B2 = np.array_split(A1, 2, axis=0)
B3, B4 = np.array_split(A2, 2, axis=0)
M1, M2, M3, M4 = B1, B2, B3, B4
print(M1.shape, M2.shape, M3.shape, M4.shape)

###### TEST ######
I, L, M = I1, L, M1
m, n, d = I.shape
##################

# Compute unit-normals from images
N = compute_normals(I, L)       # TODO(you): include your function

# Compute gradients from normals
#
# NOTE: There's no "-" in the definition of DFDY because row indexing of
#       matrices in MATLAB is reversed (row 1 = top) and so the sign must
#       be flipped in order for the y-gradients to be oriented correctly

DFDX = -N[:, :, 0] / N[:, :, 2]
DFDY =  N[:, :, 1] / N[:, :, 2]
DFDX[np.isnan(DFDX)] = 0         # Clean data
DFDX[M==0] = 0                   # Apply mask
DFDY[np.isnan(DFDY)] = 0         # Clean data
DFDY[M==0] = 0                   # Apply mask
print(DFDX.shape)
print(DFDY.shape)

# Construct least-squares problem from gradients
A = first_diffs_2d_matrix(m, n)     # TODO(you): include your function
b = A @ np.random.rand(n*m, 1)  # TODO(you): form b
# b = DFDX  # TODO(you): form b

# Built-in solver
fxy = lsqr(A, b, iter_lim=100)[0]

# Format surface
FXY = np.reshape(fxy, [m, n], order='F')         # Reshape into matrix
FXY = FXY - np.amin(FXY[M])            # Anchor to z-axis
FXY = FXY * M                     # Apply mask

fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(DFDX, cmap='gray')
axes[0, 0].axis('off')
axes[0, 1].imshow(DFDY, cmap='gray')
axes[0, 1].axis('off')
axes[1, 0].imshow(FXY, cmap='gray')
axes[1, 0].axis('off')

# Set up surface plot axes
X = np.arange(n)
Y = np.arange(m)
X, Y = np.meshgrid(X, Y)

# Plots
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
ax.imshow(DFDX, cmap='gray')
ax.axis('off')
ax = fig.add_subplot(2, 2, 2)
ax.imshow(DFDY, cmap='gray')
ax.axis('off')
ax = fig.add_subplot(2, 2, 3)
ax.imshow(FXY, cmap='gray')
ax.axis('off')
ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot_surface(X, Y, FXY, linewidth=0, antialiased=False, cmap='gray')
