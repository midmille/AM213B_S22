"""
question1.py

Author : Miles D. Miller University of California Santa Cruz 
Created : April 5, 2022 2:50pm

This file is for the implementation of the shooting method for the solution of Question 1 on the AM 213B midterm. 
The implementation follows the chapter 7 lecture notes for this form of problem.
"""
## [User Modules.]
import ODE_Solvers as OS

## [External Modules.]
import numpy as np
import matplotlib.pyplot as plt


def Ana_Y(x):
    """
    The analyticaly solved y solution. 
    """

    return (1/360)*x**6 - (1/90)*x**3 + (1/120)*x**2

def Dydx(y, x): 
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
    dydx: 1-D Array [4] 
        The solution vector for the ODEs. 

    """

    ## [The length of y.]
    N = len(y)
    ## [The empty dydt array.]
    dydx = np.zeros(N)

    ## [The rhs of the four ODE system.]
    dydx[0] = y[1]
    dydx[1] = y[2]
    dydx[2] = y[3] 
    ## [This is q(x) for the proplem.]
    dydx[3] = x**2

    return dydx


def Dydx_J(y, x): 
    """
    This is the function for the evolution equations of the Jacobian of question 1. 

    """

    ## [The dydx for the dderivatives with regards to v1 and v2.]
    dydx = np.zeros(len(y))

    ## [The compoenents of the y vector derivatives.]
    dydx[0] = y[1]
    dydx[1] = y[2]
    dydx[2] = y[3] 
    dydx[3] = 0.0

    return dydx


def Solve_J(N, dx, x, Dydx_J):
    """
    The function to solve for the jacobian of the Error. 

    Alot of the math was done analyticaly but the calculaiton of the jacobian at L=1 is done numericaly.

    """

    ## [Solving the v1 system.]
    ## [The v1 IVs.]
    dzdv1_0 = np.array([0, 0, 1, 0])
    ## [Using RK4 to solve this IVP.]
    dzdv1 = OS.RK4(N, dx, x, Dydx_J, dzdv1_0) 
    
    ## [Solving the v2 system.]
    ## [The v2 IVs.]
    dzdv2_0 = np.array([0, 0, 0, 1])
    ## [Using RK4 to solve this IVP.]
    dzdv2 = OS.RK4(N, dx, x, Dydx_J, dzdv2_0) 

    ## [Construct the jacobian matrix.]
    ## [The rows corresponds to x, and the column to the vector compenent.]
    ## [Thus the -1 index gives us the solution at x=L.]
    J = np.array([[dzdv1[-1,0], dzdv2[-1,0]], 
                  [dzdv1[-1,1], dzdv2[-1,1]]])

    return J


def Shoot_Q1(N, dx, x, Dydx, bcs, v_0, J, N_shots): 
    """
    Implementing the shooting method for the solution to question 1 system. 
    """ 

    ## [The given boundary conditions.]
    y_0, y_L, dydx_0, dydx_L = bcs
    ## [Init vi to be the v_0 guess]
    vi = v_0
    
    ## [Loop over the number of shots, although for this problem the number of shots should be 
    ##  only 1.]
    for k in range(N_shots):
        ## [The initial value guesses.]
        vi1, vi2 = vi

        ## [The initial values of the system as a guess.]
        y0 = np.array([y_0, dydx_0, vi1, vi2])

        ## [Solving the IVP.]
        y = OS.RK4(N, dx, x, Dydx, y0)

        ## [Calculate the boundary value error as a fucntion of v01, v02.]
        ## [The rows are x-coordinates, and the columns are the vector compoenents of y.]
        E = np.array([y[-1,0] - y_L, y[-1,1] - dydx_L])

        ## [Solving a linear system for vi1, vi2 of the current shot.]
        vi = vi - np.linalg.solve(J, E)

    return y 

def Plot_y(x, y, y_true): 
    """
    Plot the resulting y solution for the beam
    """

    fig, ax = plt.subplots()

    ax.plot(x, y, 'r', lw=2, label="Numerical Solution")
    ax.plot(x, y_true, 'k--',  label="Analytical Solution")
    ax.legend()
    ax.grid()
    ax.set_ylabel("y(x)")
    ax.set_xlabel("x")
    ax.set_title("Euler-Bernoulli Beam Solution")

    fig.show()

    return 


def Plot_e(x, y_num, y_ana): 
    """
    This is for plotting the error of the method.
    """

    ## [The absolute difference between the two methods]
    e = np.abs(y_ana - y_num)

    fig, ax = plt.subplots()

    ax.semilogy(x, e)
    ax.set_ylabel(r"$\mathrm{log}( | y(x_k) - u(x_k) |)$")
    ax.set_xlabel(r"$x_k$")
    ax.set_title("Absolute Bias Between Analytical and Numerical Solutions")
    ax.grid()

    fig.show()

    return 
    

if __name__ == '__main__': 
    
    ## [Some parameters given for this problem.]
    ## [Length of the beam.]
    L = 1
    ## [The number of grid points.]
    N = 60000
    Np1  = N+1
    ## [The grid spacing.]
    dx = L/N 
    ## [The grid is of length Np1.]
    x = np.linspace(0,L,Np1)
    ## [The init guess for vi.]
    v_0 = np.array([1.0, 1.0])
    ## [The  number of shots.]
    N_shots = 2

    ## [The boundary conditions for the original system.]
    y_0 = 0 
    y_L = 0
    dydx_0 = 0 
    dydx_L = 0
    bcs = [y_0, y_L, dydx_0, dydx_L]

    ## [The jacobian of the error array we seek to send to zero. This is used for the 
    ##  Newton's iterations in the shoooting method.]
    J = Solve_J(Np1, dx, x, Dydx_J)

    ## [Implementing the shooting method for the solution of this BVP.]
    y = Shoot_Q1(Np1, dx, x, Dydx, bcs, v_0, J, N_shots)
    ## [The analytical solution.]
    y_true = Ana_Y(x)

    ## [Plot the solution.] 
    Plot_y(x, y[:,0], y_true)

    ## [Plotting the error of the method.]
    Plot_e(x, y[:,0], y_true)

