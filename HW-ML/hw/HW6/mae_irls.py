import numpy as np

def mae_irls(A, b, x0, numIter):
    r"""
    Syntax:       xhat = mae_irls(A, b, x0, numIter)

    Inputs:
    -------
        A is an (m x n) matrix 
       
        b is a vector of length m

        x0 is the initial starting vector (of length n) to use
        
        numIter is the number of iterations to perform
        
    Outputs:
    -------
        xhat is a vector of length n containing the approximate solution
    """
    x = np.linalg.pinv(A)@b
    p = 1           # norm type

    for i in range(numIter):
        e = A @ x - b 				
        w = np.abs(e) ** ((p-2)/2)         # Error weights for IRLS
        W = np.diag(w / np.sum(w)) 		
        WA = W @ A 				
        x = np.linalg.solve(WA.T @ WA, WA.T @ W @ b)          # weighted L_2 sol.	
    
    return x

