##code 7(c)
import numpy as np
import random


def conv(h, x):
    y = []
    for i in range(len(h) + len(x) - 1):
        sum_temp = 0
        for j in range(len(x) - 1, -1, -1):
            if (i - j >= 0) and (i - j <= len(h) - 1):
                sum_temp += h[i - j] * x[j]
        y = np.append(y, sum_temp)
    return y.astype(int)


## examples
h = np.random.randint(10, size=random.randint(3, 10))
x = np.random.randint(10, size=random.randint(3, 10))
# h = np.array([1, 2, 3, 4, 5, 6])
# x = np.array([1, 2, 1])
print("h[]=", h)
print("x[]=", x)

print("np.convole fouction:")
print("y[]=", np.convolve(h, x))
print("my function:")
print("y[]=", conv(h, x))
print()
print()
