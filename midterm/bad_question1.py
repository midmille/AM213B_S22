"""
bad_question1.py

Author : Miles D. Miller University of California Santa Cruz 
Created : April 5, 2022 12:15pm

This file is for the implementation of the shooting method for the solution of Question 1 on the AM 213B midterm. 
This implementation was before I read the lecture notes on using the explicitly derived Jacobian.

"""

## [User Modules.]
import ODE_Solvers as OS

## [External Modules.]
import numpy as np
from scipy.optimize import newton as Newton

def Dydt(y, x): 
    """
    This is the rhs of the problem. The fourth order ODE is separated into f4 first order ODEs.

    Parameters
    ----------
    y: 1-D Array [4]
        The y-vector, where each value in the vector corresponds to the respective ODE. The length is 
        four given by the nature of the problem in Question 1. 
    x: Float
        The x-coordinate. 

    Returns 
    -------
    dydt: 1-D Array [4] 
        The solution vector for the ODEs. 

    """

    ## [The length of y.]
    N = 4 
    ## [The empty dydt array.]
    dydt = np.zeros(N)

    ## [The rhs of the four ODE system.]
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = y[3] 
    ## [This is q(x) for the proplem.]
    dydt[3] = x**2

    return dydt

def Q1_Shoot(N, dx, x, dydt, bcs, alpha_0): 
    """

    """
    
    y_0, y_L, dydt_0, dydt_L = bcs


    def E(alpha): 
        """
        The error function for use in the newton's method

        Parameters
        ----------
        alpha: 1-D Array, [2]
            The boundary condition we are solving for, two conditions. 

        Returns
        -------
        e: 1-D Array, [2]
            The resulting error from solving the initial value problem with the given 
            alpha guesses. There are two resulting error metrics.

        """

        ## [The empty error array.]
        e = np.zeros(2)

        ## [The initial values for the IVP.]
        y0_ivp = np.zeros(4)

        ## [The IVs]
        y0_ivp[0] = y_0
        y0_ivp[1] = dydt_0
        y0_ivp[2] = alpha[0]
        y0_ivp[3] = alpha[1]

        ## [Solving the initial value problem.]
        y = OS.RK4(N, dx, x, dydt, y0_ivp)

        ## [The error of the solved for BCs using provided alpha]
        e[0] = y[-1,0] - y_L
        e[1] = y[-1,1] - dydt_L

        return E

    ## [The actual shooting of the method take splaces in the Newton's solver.]
    alpha = Newton(E, alpha_0)

    return alpha




if __name__ == '__main__': 
    
    ## [Some parameters given for this problem.]
    ## [Length of the beam.]
    L = 1
    ## [The number of grid points.]
    N = 6000
    Np1  = N+1
    ## [The grid spacing.]
    dx = L/N 
    ## [The grid is of length Np1.]
    x = np.linspace(0,L,Np1)
    ## [init gues for alpha]
    alpha_0 = np.array([1,1])


    ## [The boundary conditions.]
    y_0 = 0 
    y_L = 0
    dydt_0 = 0 
    dydt_L = 0
    bcs = [y_0, y_L, dydt_0, dydt_L]

    alpha = Q1_Shoot(N, dx, x, Dydt, bcs, alpha_0)

