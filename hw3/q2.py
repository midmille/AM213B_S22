"""
Author : Miles D. Miller University of Califrnia Santa Cruz
Created : 4:16pm May 17, 2022
About: This file is for the implementation of Homework 3 Question 2 for AM213B 
       This involves the use of finite difference and collocation methods to solve the heat equation.
"""

import numpy as np 
import matplotlib.pyplot as plt
from math import sin, pi, cos


def Create_RHS_FD(Nx, u, ux01, ux02, c): 
    """
    Form the finite difference vector on the right hand side. 
    """

    ## [The empty array for the rhs.]
    u_r = np.zeros(Nx)

    ## [loop over the x grid.]
    for j in range(1,Nx-1):
        if j==1: 
            ## [This implements the lower boundary condition.]
            u_r[j] = 2*c*u[j-1] + (1-2*c)*u[j] + c*u[j+1]
        elif j==Nx-2: 
            ## [This implements the uppper boundary condition.]
            u_r[j] = c*u[j-1] + (1-2*c)*u[j] + 2*c*u[j+1]
        else: 
            u_r[j] = c*u[j-1] + (1-2*c)*u[j] + c*u[j+1]

    return u_r

def Create_LHS_A_FD(Nx, c):
    """
    Form the left hand side matrix
    """
    A = np.zeros((Nx,Nx))

    ## [Loop the rows of the matrix.]
    for j in range(Nx): 
        if j==0: 
            A[j,j] = (1+2*c)
            A[j,j+1] = -c
        elif j==Nx-1: 
            A[j,j-1] = -c 
            A[j,j] = (1+2*c)
        else: 
            A[j,j-1] = -c 
            A[j,j] = (1+2*c)
            A[j,j+1] = -c

    return A

def Solve_FD_Heat(Nx, Nt, dx, dt, x, t, ux01, ux02, ut0): 
    """
    This function solves for the solution U of the heat equation provided in question 2. 

    The system is solved mainly pen on paper for a reduced niumerical scheme. The details 
    of this will be outlined in the homework 3 report. 

    Parameters
    ----------
    Nx: Integer
        The number of grid cells in x to loop over. 
    Nt: Integer
        The number of time steps. 
    dx: Float
        The x grid spacing. 
    dt: Float
        The timestep. 
    x: 1-D Array, [Nx]
        The x grid. 
    t: 1-D Array, [Nt]
        The t-grid. 
    ux01: Float
        The x grid lower boundary condition, i.e. what u equals for all time at the smaller x boundary. 
    ux02: Float
        The x grid upper boudnary condition for all time. 
    ut0: Callable
        The function for the IV of every x position at t=0. 

    Returns 
    -------
    U: 2-D Array
        The resulting solution for u. 

    """
    
    ## [The solution matrix. Time is rows, x is columns.]
    U = np.zeros((Nt, Nx))

    ## [Constant of the scheme.]
    c = dt/(2*dx**2)

    ## [Make A matrix, its independent of time.]
    A = Create_LHS_A_FD(Nx-2, c) 

    ## [Invert the matrix.]
    Ainv = np.linalg.inv(A)

    ## [Use the initial value of the problem to set u at t=0
    U[0,:] = ut0(x)

    ## [Loop over the time.]
    for k in range(Nt-1): 
        ## [Form the rhs vector of the linear system for given time step.]
        u_r = Create_RHS_FD(Nx, U[k,:], ux01, ux02, c)
        
        ## [Solve the linear system.]
        U[k+1,1:-1] = Ainv @ u_r[1:-1]
        ## [Apply the BCS]
        U[k+1,0] = U[k,0]
        U[k+1,-1] = U[k,-1]

    return U


def Solve_Heat_GCL_Colloc(): 
    """
    This function solves the provided heat equation of homework 3 using the 
    Gauss-Chebyshev-Lobatto collocation method. 
    """

    return 

def Solve_Heat_Analytical(Nx, Nt, x, t, Ns=100): 
    """
    This solves for the analytical solution to the heat equation

    The pen on paper solution is outlined in the report for hw3. 
    
    Parameters
    ----------
    Nx: Integer
        N x nodes. 
    Nt: Integer
        N t nodes. 
    x: 1-D Array, [Nx]
        xgrid.
    t: 1-D Array, [Nt]
        tgrid.
    Ns: optional, Integer
        Default is 100. The number of values in the series solution.
    """

    def Bn(k):
        """
        """
        ## [numerator.]
#        bn_num = (5)*(-24*pi*k*sin(pi*k) - (24-8*pi**2*k**2)*cos(pi*k) - pi**4*k**4 - 4*pi**2*k**2 - 24)
#        bn_num = (pi**4*k**4 + 4*pi**2*k**2 + 24 - 24*pi*k*sin(pi*k) - (24 - 8*pi**2*k**2)*cos(pi*k))
#        bn_num = (2*pi**4*k**4 + 32*pi**2*k**2 + 768)*cos((pi*k)/2) - 384*pi*k*sin(pi*k) - (768 - 64*pi**2*k**2)*cos(pi*k)  
#        bn_num = 320*(6*pi*k*sin(pi*k) + (12-pi**2*k**2)*cos(pi*k) + pi**2*k**2 - 12) 
        bn_num = 160*(6*pi*k*sin(pi*k) + (12-pi**2*k**2)*cos(pi*k) + pi**2*k**2 - 12)
        ## [The denominator.]
        bn_den = pi**5*k**5

        ## [bn sol.]
        bn = -((bn_num)/(bn_den))

        return bn

    ## [Make a mesh grid of x and t.]
    xx, tt = np.meshgrid(x, t) 

    ## [The empty solution for U.]
    ## [Rows are t, columns are x.]
    nu = np.zeros((Nt,Nx))
    U  = np.zeros((Nt,Nx))

    ## [The series solution to nu.]
    for k in range(1,Ns): 
        nu = nu + 2*Bn(k) * np.sin(k*np.pi*((xx+1)/2))*np.exp(-(k**2*np.pi**2*tt)/4)

    ## [Reconstruct the system.]
    U = (3+xx) + nu

    return U


def Plot_Usol(x, t, U): 
    """
    This plots the resulting U solution as a surface plot
    """

    ## [Mesh of a x and t, recall, t is the rows, x is columns.]
    xx, tt = np.meshgrid(x, t)

    ## [Plotting.]
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

    im = ax.plot_surface(tt, xx, U)

    fig.show()

    return 

if __name__ == "__main__": 

    ## [The list of grid points for the error analysis.]
    Ns= [10,20,40,60,80,100,110,120,130,140,150]
    
    err = np.zeros(len(Ns))

    ## [Loop the Ns]
    for k, Nx in enumerate(Ns):
        ## [Some problem parameters.]
        ## [xgrid params.]
        x = np.linspace(-1, 1, Nx)
        dx = x[0] - x[1]
        ## [tgrid params.]
        dt = 1e-4
        t = np.arange(0.0, 2.0+dt, dt)
        Nt = len(t)
        ## [The temporal initial value.]
        ut0 = lambda x: (3+x) + 5*(1-x**2)**2
        ## [The spatial boundary conditions.]
        ## [Lower bc at x = -1.]
        ux01 = 2
        ## [Upper bc at x =1.]
        ux02 = 4
    
        ## [Solve for U using finite difference and crank nicoloson.]
        Ufd = Solve_FD_Heat(Nx, Nt, dx, dt, x, t, ux01, ux02, ut0)
    
        ## [Solve for the Analytical solution to U.]
        Uan = Solve_Heat_Analytical(Nx, Nt, x, t, Ns=100)

        diff = np.absolute(Ufd - Uan)

        err[k] = np.max(diff[-1, :])

    fig, ax = plt.subplots()

    ax.semilogy(Ns, err)
    ax.set_xlabel('Nx')
    ax.set_ylabel('e(t)')
    ax.grid()
    ax.set_title('Error of the Finite Difference Solution')

    fig.show()


    
    ## [Plot the solution.]
#    Plot_Usol(x, t, Ufd)
#    Plot_Usol(x, t, Uan)
    
    
