"""

Author: Miles D. Miller UCSC 
Created: 12:32pm April 10, 2022 

This file contains ODE schemes.
"""

import numpy as np


def RK3(N, dt, t, dydt, y0)
    """
    This function implements an explicit 3 stage Runge Kutta method for the solution
    of the given ODE. This is used for the solution to Homework 1, Question 1, Part B, Section 2
    in 213B. 

    The algorithm assumes a uniform grid. 

    It also assumes that y is a vector. 

    Parameters
    ----------
    N: Integer
        The number of grid points. 
    dt: Float
        The time step. 
    t: 1-D Array, [N]
        The uniform time grid. 
    dydt: Function
        The rhs side of the differential equation. 
    y0: 1-D Array, [M]
        The initial condition of the IVP. 

    Returns 
    -------
    y: 2-D Array, [N, M]
        The solution to the ODE, with y as a vector. 
        
    """

    ## [The length of y0 tells us how many vectors there are]
    M = len(y0)

    ## [Init the solution y array.]
    y = np.zeros((N,M))

    ## [Set the initial conditions.]
    y[0,:] = y0

    ## [Loop over the t-grid.]
    for k in range(N-1): 
        
        ## [The three stages of RK3]
        k1 = dydt(y[k,:], t[k])
        k2 = dydt(y[k,:] + dt*0.5*k1, t[k] + 0.5*dt)
        k3 = dydt(y[k,:] + dt*(-1k1 + 2k2), t[k] + dt)

        ## [Solving for the next step in y.]
        y[k+1, :] = y[k,:] + dt*((1/6)*k1 + (2/3)*k2 + (1/6)*k3)

    return y
    

        
 
