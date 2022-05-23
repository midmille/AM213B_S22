""" 
Author : Miles D. Miller University of Califrnia Santa Cruz
Created : 6:33pm May 20, 2022
About: This file is for the implementation of Homework 3 Question 3 for AM213B 
       It involves the implementation of Kuramoto-Sivashinsky initial-boundary problem using finite
       difference and Adams-Bashforth two-step methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 

def CFD_1(N, dx, u): 
    """
    Implements second order central finite difference for the first derivative of provided
    u-vector. The boundary conditions implemented are periodic. 

    Parameters
    ----------
    N: Integer
        The number of grid points.
    dx: Float
        The grid cell spacing.
    u: 1-D Array, [N]
        The vector of array values to solve for the derivative of. 

    Returns
    -------
    dudx: 1-D Array, [N]
        The finite difference derivative vector solution.
    """

    ## [Empty solution vector.]
    dudx = np.zeros(N)

    ## [Loop over the x nodes.] 
    for k in range(N): 
        ## [The periodic boundary conditions.]
        if k == 0: 
            dudx[k] = (u[k+1] - u[N-1])/(2*dx)
        elif k == N-1: 
            dudx[k] = (u[0] - u[k-1])/(2*dx)
        ## [Rest of the domain.]
        else: 
            dudx[k] = (u[k+1] - u[k-1])/(2*dx)

    return dudx


def CFD1_mat(N, dx): 
    """
    This formulates the first derivative CFD matrix for periodic BCs. 
    """

    ## [Empty A matrix.]
    A = np.zeros((N,N))
    ## [Loop over the N]
    for k in range(N): 
        if k == 0: 
            A[k,N-1] = -1
            A[k,k+1] = 1
        elif k == N-1: 
            A[k,k-1] = -1
            A[k, 0] = 1
        else: 
            A[k,k-1] = -1
            A[k,k+1] = 1

    return (1/(2*dx)) * A
        

def CFD_2(N, dx, u): 
    """
    Implements second order central finite difference for the second derivative of provided
    u-vector. The boundary conditions implemented are periodic. 

    Parameters
    ----------
    N: Integer
        The number of grid points.
    dx: Float
        The grid cell spacing.
    u: 1-D Array, [N]
        The vector of array values to solve for the derivative of. 

    Returns
    -------
    d2udx2: 1-D Array, [N]
        The finite difference derivative vector solution.
    """

    ## [Empty solution vector.]
    d2udx2 = np.zeros(N)

    ## [Loop over the x nodes.] 
    for k in range(N): 
        ## [The periodic boundary conditions.]
        if k == 0: 
            d2udx2[k] = (u[N-1] - 2*u[k] + u[k+1])/(dx**2)
        elif k == N-1: 
            d2udx2[k] = (u[k-1] - 2*u[k] + u[0])/(dx**2)
        ## [Rest of the domain.]
        else: 
            d2udx2[k] = (u[k-1] - 2*u[k] + u[k+1])/(dx**2)

    return d2udx2


def CFD2_mat(N, dx): 
    """
    This formulates the second derivative CFD matrix for periodic BCs. 
    """

    ## [Empty A matrix.]
    A = np.zeros((N,N))
    ## [Loop over the N]
    for k in range(N): 
        if k == 0: 
            A[k,N-1] = 1
            A[k,k] = -2
            A[k,k+1] = 1
        elif k == N-1: 
            A[k,k-1] = 1
            A[k,k] = -2
            A[k, 0] = 1
        else: 
            A[k,k-1] = 1
            A[k,k] = -2 
            A[k,k+1] = 1

    return (1/(dx**2)) * A


def CFD_4(N, dx, u): 
    """
    Implements second order central finite difference for the fourth derivative of provided
    u-vector. The boundary conditions implemented are periodic. 

    Parameters
    ----------
    N: Integer
        The number of grid points.
    dx: Float
        The grid cell spacing.
    u: 1-D Array, [N]
        The vector of array values to solve for the derivative of. 

    Returns
    -------
    d4udx4: 1-D Array, [N]
    """

    ## [Empty solution vector.]
    d4udx4 = np.zeros(N)

    ## [Loop the x nodes.]
    for k in range(N): 
        if k == 0: 
            d4udx4[k] = (u[k+2] - 4*u[k+1] + 6*u[k] - 4*u[N-1] + u[N-2]) / (dx**4)
        elif k == 1: 
            d4udx4[k] = (u[k+2] - 4*u[k+1] + 6*u[k] - 4*u[k-1] + u[N-1]) / (dx**4)
        elif k == N-1: 
            d4udx4[k] = (u[1] - 4*u[0] + 6*u[k] - 4*u[k-1] + u[k-2]) / (dx**4)
        elif k == N-2: 
            d4udx4[k] = (u[0] - 4*u[k+1] + 6*u[k] - 4*u[k-1] + u[k-2]) / (dx**4)
        else: 
            d4udx4[k] = (u[k+2] - 4*u[k+1] + 6*u[k] - 4*u[k-1] + u[k-2]) / (dx**4)

    return d4udx4


def CFD4_mat(N, dx): 
    """
    This formulates the second derivative CFD matrix for periodic BCs. 
    """

    ## [Empty A matrix.]
    A = np.zeros((N,N))
    ## [Loop over the N]
    for k in range(N): 
        if k == 0: 
            A[k, N-2] = 1
            A[k,N-1] = -4
            A[k,k] = 6
            A[k,k+1] = -4
            A[k,k+2] = 1
        elif k == 1: 
            A[k,N-1] = 1
            A[k,k-1] = -4
            A[k,k] = 6
            A[k,k+1] = -4
            A[k,k+2] = 1
        elif k == N-1:
            A[k, k-2] = 1
            A[k,k-1] = -4
            A[k,k] = 6
            A[k,0] = -4
            A[k,1] = 1
        elif k==N-2: 
            A[k, k-2] = 1
            A[k,k-1] = -4
            A[k,k] = 6
            A[k,k+1] = -4
            A[k,0] = 1
        else: 
            A[k, k-2] = 1
            A[k,k-1] = -4
            A[k,k] = 6
            A[k,k+1] = -4
            A[k,k+2] = 1

    return (1/(dx**4)) * A


def Solve_KS_IBP(Nx, Nt, dx, dt, x, t, ut0): 
    """
    This fucntion solves for the solution to the Kuramoto-Sivashinsky initial-boundary value problem provided
    by question 3 of homework 3 for AM 213B.

    The time stepping is done using Adams-Bashforth two step method.

    The boundary conditions are periodic.

    Parameters
    ----------
    Nx: Integer
        The number of grid points in x.
    Nt: Integer
        The number of grid points in t. 
    dx: Float
        The x grid spacing. 
    dt: Float
        The t grid spacing. 
    x: 1-D Array, [Nx]
        The x grid vector. 
    t: 1-D Array, [Nt]
        The t grid vector.
    ut0: Callable
        The function for u at t=0. 

    Returns
    -------
    U: 2-D Array, [Nt, Nx]
        The solution array for u in time and space. Time is the rows and x is the columns.
    """

    def F(Nx, dx, u): 
        """
        The function representing the rhs of the time stepping problem. This is the finite difference 
        operators on the rhs.
        """
#        return -u*CFD_1(Nx, dx, u) - CFD_2(Nx, dx, u) - CFD_4(Nx, dx, u)
        return -u*(CFD1@u) - (CFD2@u) - (CFD4@u)


    CFD1 = CFD1_mat(Nx, dx)
    CFD2 = CFD2_mat(Nx, dx)
    CFD4 = CFD4_mat(Nx, dx)

    ## [The empty U array.]
    U = np.zeros((Nt, Nx))

    ## [Initialize U with the initial value function.]
    U[0,:] = ut0(x)

    ## [Perform RK 3 to Init the U[1,:] vector before AB multistep method.]
#    k1 = F(Nx, dx, U[0,:])
#    k2 = F(Nx, dx, U[0,:] + dt*0.5*k1)
#    k3 = F(Nx, dx, U[0,:] + dt*(-1*k1 + 2*k2))
#    U[1,:] = U[0,:] + dt*((1/6)*k1 + (2/3)*k2 + (1/6)*k3)
    k1 = F(Nx, dx, U[0,:])
    k2 = F(Nx, dx, dt*k1)
    U[1,:] = U[0,:] + (dt/2)*(k1 + k2)

    ## [Loop over the time.]
    for k in range(1,Nt-1): 
        U[k+1,:] = U[k,:] + (dt/2) * (3*F(Nx, dx, U[k,:]) - F(Nx, dx, U[k-1, :]))

    return U


def Plot_Usol(x, t, U): 
    """
    This plots the resulting U solution as a surface plot
    """

    ## [Mesh of a x and t, recall, t is the rows, x is columns.]
    xx, tt = np.meshgrid(x, t)

    ## [Plotting.]
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

    im = ax.plot_surface(tt, xx, U, cmap ='viridis')
    ax.set_ylabel('x')
    ax.set_xlabel('t')
    ax.set_zlabel('U(x, t)')

    fig.show()

    ## [Plot Color map solution.]
    fig, ax = plt.subplots()

    im = ax.pcolormesh(tt, xx, U, cmap='nipy_spectral')
    ax.set_ylabel('x')
    ax.set_xlabel('t')

    plt.colorbar(im, ax=ax)

    fig.show()

    return 


if __name__ == '__main__':
    
    ## [Problem parameters.]
    Nx = 200
    dx = 60/(Nx)

    ## [The x grid.]
    x = np.arange(-30, 30 ,dx)
    Nx = len(x)
    ## [The time grid.]
    dt = 5e-4
    t = np.arange(0, 30, dt)
    Nt = len(t)

    ## [The initial condition function.]
    ut0 = lambda x: np.exp(-x**2)

    ## [Solving for U.]
    U = Solve_KS_IBP(Nx, Nt, dx, dt, x, t, ut0)

    ## [Save the result.]
    pickle.dump(U, open('q3_Usol.p', 'wb'))

    ## [Plotting the solution.]
    Plot_Usol(x, t, U)
