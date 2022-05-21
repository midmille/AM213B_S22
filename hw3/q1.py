"""
Author : Miles D. Miller University of Califrnia Santa Cruz
Created : 4:16pm May 17, 2022
About : This file is for Question 1, Homework 3 in AM 213B.
"""

import numpy as np

def a(x): 
    return 2.0 - x**2


def dadx(x): 
    """
    The derivative of a, solved analytically. 
    """
    return -2*x


def b(x): 
    return np.sin(np.pi*x)


def f(x): 
    return np.exp(-x)

def Create_Q1_Sys(N, dx, x, alpha, beta):
    """
    This creates the matrix system for question one. 
    
    This includes the boundary conditions. 

    The majority of this has been solved for on paper, see latex solution report for
    the work. 
    """

    A = np.zeros((N,N))
    y = np.zeros(N)

    ## [Loop over the nodes.]
    for k in range(N): 
        
        ## [Setting up the LHS A matrix.]
        ## [u_0 Neumann boundary condition.]
        if k == 0: 
            ## [Implement forward finite difference.]
            A[k, k] = -3*dx
            A[k, k+1] = 4*dx
            A[k, k+2] = -dx
        ## [The derishle boundary condition.] 
        elif k == N-1: 
#            A[k, k-1] = -(4*a(x[k]) + 2*(dx**2)*b(x[k]))
#            A[k, k] = (dx*dadx(x[k]) + 2*a(x[k]))
            A[k, k] = 2*dx**2
        ## [Rest of the matrix. ]
        else: 
            A[k, k-1] = -(dx*dadx(x[k]) + 2*a(x[k]))
            A[k, k]   = (4*a(x[k]) + 2*(dx**2)*b(x[k]))
            A[k, k+1] = (dx*dadx(x[k]) - 2*a(x[k]))

        ## [Setting up the RHS solution vector.]
        if k == 0: 
            y[k] = alpha
        elif k == N-1: 
            y[k] = beta
        else: 
            y[k] = f(x[k])

    A = (1/(2*dx**2)) * A

    return A, y


if __name__ == '__main__': 
    
    ## [The number of points.]
    N = 500 
    x = np.linspace(-1, 1, N)
    dx = x[0] - x[1]

    ## [Boundary params.]
    alpha = 0.0
    beta = 2.0

    A, y = Create_Q1_Sys(N, dx, x, alpha, beta)

    u = np.linalg.solve(A,y)
