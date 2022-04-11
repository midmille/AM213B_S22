"""

Author: Miles D. Milles UCSC
Created: 12:40pm April 8, 2022

This file is for Question 1 Part B Section 2 on Homework 1 for AM 213B

"""
## [External Modules.]
import numpy as np
import matplotlib.pyplot as plt
## [Internal Modules.]
from ODE_Solvers import RK3

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

    ## [Size of grid.]
    N = len(t)

    ## [Init y.]
    y = np.zeros((N, 2))

    ## [The y1 solution.]
    y[:, 0] = np.exp(-3*t) * (np.sin(t) - 3*np.cos(t))
    ## [The y2 solution.]
    y[:,1] = np.exp(-3*t) * (np.cos(t) + 3*np.sin(t))

    return y

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

    dydt = np.dot(A, y)

    return dydt
    

def AM2_1b2_spec(dydt, A, N, dt, t, y0): 
    """
    This implements the Implicit Adams Moulton derived for problem q1b2. 

    Parameters
    ----------
    N: Integer
        The number of grid points. 
    dt: Float
        The time step. 
    t: 1-D Array, [N]
        The uniform time grid. 
    y0: 1-D Array, [M]
        The initial condition of the IVP. 

    Returns 
    -------
    y: 2-D Array, [N, M]
        The solution to the ODE, with y as a vector. 
    """

    ## [The number of y coordinates in the system]
    M = len(y0)

    ## [Init the y.]
    y = np.zeros((N,M))

    ## [Set the initial value condition.]
    y[0,:] = y0 

    ## [A shaped Idenity matrix.]
    I = np.array([[1,0], 
                  [0,1]])
    ## [Loop t.]
    for k in range(N-1): 
        
        ## [Boundary condition is that u_k-1 = u_k at initial value.]
        if k == 0: 
            ## [Using an order 3 explicit to get y[1, :]. 
            ## [The three stages of RK3]
            k1 = dydt(y[k,:], t[k])
            k2 = dydt(y[k,:] + dt*0.5*k1, t[k] + 0.5*dt)
            k3 = dydt(y[k,:] + dt*(-1*k1 + 2*k2), t[k] + dt)

            ## [Solving for the next step in y.]
            y[k+1, :] = y[k,:] + dt*((1/6)*k1 + (2/3)*k2 + (1/6)*k3)
        else:
            ## [This is Adam Moulton solution is derived in the hw pdf.]
            ## [y_k+1 = (1-dt/12 *5*a)^-1 (u_k + dt/12(8*A*u_k - A*u_k-1)) ] 

            y[k+1, :] = np.linalg.solve(I-(dt/12)*5*A, y[k,:] + (dt/12)*(8*np.dot(A, y[k,:]) - np.dot(A, y[k-1, :])))
    
    return y


if __name__ == '__main__': 
    

    ## [A used globaly.]
    A = np.array([[-3,1],
                  [-1, -3]])

    ## [The total time.]
    T = 10

    ## [The dt.]
    dt = 0.0005

    ## [The t grid.]
    t = np.arange(0, T, dt)
    ## [The number of grip points.]
    N = len(t)

    ## [The initial condition.]
    y0 = np.array([-3, 1])

    ## [Run the analytical solution.]
    y = analytical_y(t)

    ## [Run the RK3 solution.]
    y_rk3 = RK3(N,dt, t, dydt_1b2, y0)

    ## [Run the Adam Moulton method.]
    y_am2 = AM2_1b2_spec(dydt_1b2, A, N, dt, t, y0)

    ## [Plotting.]
    fig, ax = plt.subplots()
    ## [The analytical sol.]
    ax.plot(t, y, '-o', markersize = 1, label='Analytical Solution')
    ## [The rk3 sol.]
    ax.plot(t, y_rk3, '-o', markersize = 1, label='RK3 Solution')
    ## [The AM2 solution.]
    ax.plot(t, y_am2, '-o', markersize = 1, label='AM2 Solution')

    ax.set_ylabel('Y')
    ax.set_xlabel('t')
    ax.set_title(f'dt = {dt}')
    ax.legend()
    ax.grid()

    fig.show()

    
    
    
