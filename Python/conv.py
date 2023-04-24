import numpy as np


def conv(x, y):
    c = []
    for i in range(len(x) + len(y) - 1):
        sum_temp = 0
        for j in range(len(y) - 1, -1, -1):
            if (i - j >= 0) and (i - j <= len(x) - 1):
                sum_temp += x[i - j] * y[j]
        c = np.append(c, sum_temp)
    return c


## examples
x = np.array([1, 2, 1])
y = np.array([1, 2, 3, 4, 5, 6])
print(np.convolve(x, y))
print(conv(x, y))
