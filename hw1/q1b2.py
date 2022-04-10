"""

Author: Miles D. Milles UCSC
Created: 12:40pm April 8, 2022

This file is for Question 1 Part B Section 2 on Homework 1 for AM 213B

"""

import numpy as np

def analytical_y(t): 
    """
    The analyticaly solved solution for the problem. 

    In this case: 

    y0= [-3, 1]^T
    
    A = [[-3, 1], 
         [-1, -3]]

    The solution is derived in the homework pdf. 

    Mainly consist of solving for eigen values and finding the real component of the 
    general equation for a second order ODE. 

    The analytical solution is as follows: 
    
    y(y) = e^(-3t) [(sin(t) - 3cost(t)), (cost(t) + 3sin(t))]^T

    Parameters
    ----------
    t: 1-D Array, [N]
        This is the time grid. 

    Returns
    -------
    y: 2-D Array, [N, 2]
        The analytical solution to this problem. 

    """

    ## [The y1 solution.

def dydt_1b2(y, t): 
    """
    This is the rhs of the ODE for question 1.b.2 on homework one. 
    
    Parameters
    ----------
    y: 1-D Array, [2]
        The vector value y, the length should be 2. 
    
    Returns 
    -------
    dydt: 1-D Array, [2]
        The vector derivative solution of the ODE. Length should be 2. 

    """

    A = np.array([[-3,1],
                  [-1, -3]])

    dydt = np.dot(A, y)

    return dydt
    

       



if __name__ == '__main__': 
    
    
