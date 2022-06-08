"""
numerical_sol.py
Author : Miles D. Miller
Created : June 8, 2022 11:01am 
About : This file is the numerical solution and the accompanying studies for the AM 213B final.
"""

import numpy as np
import matplotlib.pyplot as plt

def Create_Space_Grid(r2, r1, Np1): 
    """
    This function forms the spatial grid for the solution following equation (9) on the 
    final exam.
    """

    dr = (r2 - r1) / (Np1)

    ## [This is equivalent to r[j] = r1 + j*dr ; where j = 0,...,Np1 (including Np1)]
    r = np.arange(r1, r2+dr, dr)

    return r


def Create_M(N, nu, r, dr): 
    """
    This formulates the rhs matrix of the IBVP scheme for the final. The matrix construction 
    is outlined in the report.

    Parameters
    ----------
    N: Integer
        The number of nodes not including BC nodes, so j=1,..,N, but for r, j=0,..,Np1
    nu : Float
        Problem constant.
    r: 1-D Array, [N+2]
        This array is indexed from j=0,..,Np1 including Np1.
    dr: Float, 
        The seperation between spatial nodes.

    Returns: 
    --------
    M: 2-D Array, [N,N]
        The finite difference array for the innier nodes of the problem.
    """

    ## [The matrix will only be NxN since the BCs at j=0, j=N+1 are static and known.]
    M = np.zeros((N,N))
    
    ## [Eta only depends on dr, not r.]
    eta = 1/(dr**2)

    ## [python is zero based indexing.]
    for j in range(N):
        
        ## [Calculate the FD constant alpha, see report for deinition and derivation.]
        ## [recall that r includes the BC nodes here. Therefor must use j+1 for inner.]
        alpha = 1 / (2*dr*r[j+1])
        ## [The lower BC.]
        if j == 0: 
            M[j,j] = -2*eta 
            M[j,j+1] = eta + alpha
        ## [Upper BC.]
        elif j == N-1:
            M[j,j-1] = eta - alpha  
            M[j,j] = -2*eta 
        else:
            M[j,j-1] = eta - alpha  
            M[j,j] = -2*eta 
            M[j,j+1] = eta + alpha
    
    ## [Multiplit the matrix by the problem constant.]
    M = nu*M

    return M


def Spectral_Radius_M():
    """

    """

    return 


def Absolute_Stability(): 
    """

    """

    return


def U_num(r, t, dr, dt, nu, U0):
    """
    This solves for U numerically using finite difference and AB3. It is assumed that 
    t and r are constant grid.

    It is also assumed zero valued boundary conditions in space. 

    It is assumed that r[0] = r1 and r[Np1] = r[-1] = r2, s.t. len(r) = Np2
    """

    ## [The total length of r is Np2.]
    Nrp2 = len(r)
    Nrp1 = Nrp2 - 1
    Nr = Nrp1 - 1

    ## [Total length of time is Nt.]
    Nt = len(t)

    ## [The rows are t, the y-axis, and the columns, the x-axis, is r. ]
    U = np.zeros((Nt, Nrp2))
    
    ## [Construct the FD rhs matrix M, size NxN.]
    M = Create_M(N, nu, r, dr)

    ## [Must inittialize the first step of U to be U0.]
    U = U0(r)

    ## [Must initialize steps k=1,2 for AB3 method using a one step method of the 
    ##  same order, namely RK3.]




    return


if __name__ == '__main__':
    pass



