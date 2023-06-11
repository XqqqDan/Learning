import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from compute_normals import compute_normals

# load data
# I = images   (m x n x d)
# L = lighting (3 x d)
# M = mask     (m x n)
catData = sio.loadmat("cat.mat")
I = catData["I"]
L = catData["L"]
M = catData["M"]

N = compute_normals(I, L)

plt.figure()
plt.subplot(131)
plt.imshow(N[:, :, 0], cmap="gray")
plt.axis("off")
plt.subplot(132)
plt.imshow(N[:, :, 1], cmap="gray")
plt.axis("off")
plt.subplot(133)
plt.imshow(N[:, :, 2], cmap="gray")
plt.axis("off")
plt.show()
