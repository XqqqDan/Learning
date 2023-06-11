import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(100)
y = np.random.randn(100)
A = np.outer(x, y)
s = np.linalg.svd(A, compute_uv=False)
plt.stem(s)
plt.xlabel("Index")
plt.ylabel("Singular value")
plt.show()
