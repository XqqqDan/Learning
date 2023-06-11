import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from pylab import xticks, yticks

# 目标函数
def real_func(x):
    return 0.5*np.exp(0.8*x)

# 多项式
# ps: numpy.poly1d([1,2,3])  生成  $1x^2+2x^1+3x^0$*
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

# 十个点
x = np.linspace(0, 2, 16)
x_points = np.linspace(0, 2, 1000)
# 加上正态分布噪音的目标函数的值
y = real_func(x)
y_noise = [np.random.normal(0, 0.1)+y1 for y1 in y]

def fitting(M=0, flag=0):
    """
    M 为 多项式的次数
    """

    # 随机初始化多项式参数
    p_init = np.random.rand(M+1)
    # 最小二乘法
    if flag == 0:
        p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    else:
        p_lsq = leastsq(residuals_func, p_init, args=(x, y_noise))

    print('Degree=', M, 'polynomial:', p_lsq[0])

    return p_lsq

def compare(M1, M2, flag=0):
    """
    M1,M2为对比阶数
    """
    p_lsq1 = fitting(M1, flag)
    p_lsq2 = fitting(M2, flag)

    # 可视化
    label = 'Least square, degree='
    label1 = label + str(M1)
    label2 = label + str(M2)

    plt.figure()
    plt.plot(x, fit_func(p_lsq1[0], x), '--', linewidth=2, label=label1)
    plt.plot(x, fit_func(p_lsq2[0], x), ':',  linewidth=2, label=label2)

    if flag == 0:
        plt.plot(x, real_func(x), 'bo', label='without noise')
    else:
        plt.plot(x, y_noise, 'bo', label='with noise')

    xticks(np.linspace(0, 2, 16, endpoint=True))
    yticks(np.linspace(0, 3, 13, endpoint=True))

    plt.grid()
    plt.legend()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    compare(15, 2, 0)
    compare(15, 2, 1)
    plt.show()

