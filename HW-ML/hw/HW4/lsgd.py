import numpy as np

def lsgd(A, b, mu, x0, numIter):
    r"""
    Syntax:       xgd = lsgd(A,b,mu,x0,maxIters)

    Inputs:
    -------
        A is an (m x n) matrix 
       
        b is a vector of length m

        mu is the step size to use and must satisfy
        0 < mu < 2/norm(A)^2 to guarantee convergence
        
        x0 is the initial starting vector (of length n) to use
        
        numIter is the number of iterations to perform

    Outputs:
    -------
        xgd is a vector of length n containing the approximate solution
    """
    x = x0
    top = 2/np.linalg.norm(A)

    for i in range(numIter):
        if mu >= top:
            print('Step is out range')
            break

        x = x - mu*A.T@(A@x-b)

    xgd = x

    return xgd

if __name__ == '__main__':
    m = 100
    n = 50
    sigma = 0.1
    np.random.seed(0)

    # set up A, x, and b
    A = np.random.randn(m, n)
    xtrue = np.random.rand(n, 1)
    b = A @ xtrue + sigma * np.random.randn(m, 1)

    # parameters for lsgd
    mu = 1 / np.linalg.norm(A, ord=2) ** 2
    x0 = np.zeros((n, 1))
    numIter = 300

    print(lsgd(A, b, mu, x0, numIter))


